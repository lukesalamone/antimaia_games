import asyncio
from pathlib import Path
import subprocess
import chess
import chess.engine
import chess.pgn
from subprocess import PIPE, DEVNULL
from stockfish import Stockfish
import numpy as np
from datetime import datetime

class Player:
    def __init__(self, name):
        self.name = name

    async def move(self, board):
        raise NotImplementedError('classes must inherit from base class')

class MaiaPlayer(Player):
    def __init__(self, name, weights_path, parallel=False):
        super().__init__(name)
        self.weights_path = weights_path
        self.lc0Path = '/usr/local/bin/lc0'
        self.engine = None
        self.parallel = parallel
        self.limit = chess.engine.Limit(depth=0)

    async def move(self, fen):
        board = chess.Board(fen)

        if self.parallel:
            fen, distribution = await self._get_maia_distribution_async(fen)
        else:
            fen, distribution = self._get_maia_distribution(fen)

        if len(distribution) < 1:
            return None

        sample = True   # sample from the move distribution
        move = ''
        if not sample:
            move = distribution[0][0]
        else:
            moves, probs = zip(*distribution)
            norm = sum(probs)
            move = np.random.choice(moves, p=[x/norm for x in probs])
        return chess.Move.from_uci(move)

    def _parse_info(self, s):
        s = ''.join(s.split(' '))
        move = s.split('(')[0]
        pct = float(s.split('P:')[1].split('%')[0]) / 100
        return move, pct

    async def _call_maia_parallel(self, fens):
        fens = set(fens)
        distributions = await asyncio.gather(*[self._get_maia_distribution_async(fen) for fen in fens])
        return {fen: distribution for fen, distribution in distributions}

    def _call_maia(self, fens):
        distributions = {}
        for fen in fens:
            f, dist = self._get_maia_distribution(fen)
            distributions[fen] = dist
        return distributions

    def _get_maia_distribution(self, fen):
        engine = chess.engine.SimpleEngine.popen_uci(
            [self.lc0Path, f'--weights={self.weights_path}', '--verbose-move-stats'],
            stderr=DEVNULL, stdout=PIPE)

        infos = []
        with engine.analysis(chess.Board(fen), self.limit) as analysis:
            for info in analysis:
                if 'string' in info:
                    move, pct = self._parse_info(info['string'])
                    if move != 'node':
                        infos.append((move, pct))

        engine.quit()
        return [fen, sorted(infos, key=lambda x: x[1], reverse=True)]

    async def _get_maia_distribution_async(self, fen):
        transport, engine = await chess.engine.popen_uci(
                [self.lc0Path, f'--weights={self.weights_path}', '--verbose-move-stats'],
                stderr=DEVNULL, stdout=PIPE)

        infos = []
        with await engine.analysis(chess.Board(fen), self.limit) as analysis:
            async for info in analysis:
                if 'string' in info:
                    move, pct = self._parse_info(info['string'])
                    if move != 'node':
                        infos.append((move, pct))
        engine.quit()
        return [fen, sorted(infos, key=lambda x: x[1], reverse=True)]

class AntimaiaPlayer(MaiaPlayer):
    def __init__(self, name, weights_path, stockfish_depth, parallel=False):
        super().__init__(name, weights_path, parallel)
        self.weights_path = weights_path
        self.stockfish_depth = stockfish_depth
        self.stockfish = Stockfish(
            path='/usr/local/bin/stockfish',
            depth=stockfish_depth,
            parameters={"Threads": 2, "Hash": 32})
        self.checkmate_weight = 10 * 100      # value in centipawns
        self.parallel = parallel

    async def move(self, fen):
        return await self._get_antimaia_move(fen)

    async def _get_antimaia_move(self, fen):
        def ideal_move(mate, val, white):
            if not mate:
                return False
            if val > 0 and white:
                return True
            if val < 0 and not white:
                return True
            return False

        start_fen = fen
        board = chess.Board(start_fen)
        white = board.turn
        moves = list(board.legal_moves)
        best_move = moves[0]
        best_score = float('inf') if not white else -float('inf')

        # if stockfish sees mate, play it
        val, mate = self._get_stockfish_eval(fen)
        if ideal_move(mate, val, white):
            self.stockfish.set_fen_position(start_fen)
            return chess.Move.from_uci(self.stockfish.get_best_move())

        # gather all fens to evaluate with Maia and call Maia in parallel
        maia_fens = set()
        stockfish_fens = set()
        for move in moves:
            _board = chess.Board(start_fen)
            _board.push(move)
            _fen = _board.fen()
            maia_fens.add(_fen)
            stockfish_fens.add(_fen)

        maia_distributions = {}
        if self.parallel:
            asyncio.set_event_loop(asyncio.new_event_loop())
            maia_distributions = await asyncio.gather(self._call_maia_parallel(maia_fens))
            maia_distributions = maia_distributions[0]
        else:
            maia_distributions = self._call_maia(maia_fens)

        # gather all fens to evaluate with Stockfish and call Stockfish in parallel
        for fen, dist in maia_distributions.items():
            stockfish_fens.add(fen)
            for mv, pct in dist:
                _board = chess.Board(fen)
                _board.push(chess.Move.from_uci(mv))
                stockfish_fens.add(_board.fen())

        stockfish_evals = {}
        for fen in stockfish_fens:
            stockfish_evals[fen] = self._get_stockfish_eval(fen)

        for move in moves:
            board.set_fen(start_fen)
            board.push(move)
            val, mate = stockfish_evals[board.fen()]

            if ideal_move(mate, val, white):
                return move

            distribution = maia_distributions[board.fen()]
            skip = False
            move_results = []

            for mv, pct in distribution:
                _board = board.copy()
                _board.push(chess.Move.from_uci(mv))
                val, mate = stockfish_evals[_board.fen()]
                move_results.append(val * pct)

            score = sum(move_results)

            if white and score > best_score:
                best_score = score
                best_move = move
            elif not white and score < best_score:
                best_score = score
                best_move = move
        return best_move

    def _get_stockfish_eval(self, fen):
        self.stockfish.set_fen_position(fen)
        evaluation = self.stockfish.get_evaluation()
        if evaluation['type'] == 'mate':
            return evaluation['value'] * self.checkmate_weight, True
        else:
            return evaluation['value'], False

class StockfishPlayer(Player):
    def __init__(self, name, depth):
        super().__init__(name)
        self.depth = depth
        self.stockfish = Stockfish(
            path='/usr/local/bin/stockfish',
            depth=self.depth,
            parameters={"Threads": 2, "Hash": 32})

    async def move(self, fen):
        self.stockfish.set_fen_position(fen)
        return chess.Move.from_uci(self.stockfish.get_best_move())

class GameManager:
    def __init__(self, white_player, black_player, verbose=False, round=None):
        self.white = white_player
        self.black = black_player
        self.verbose = verbose
        self.round = round

    async def pit(self):
        board = chess.Board()
        move_count = 1

        while not board.is_game_over():
            w = await self.white.move(board.fen())
            board.push(w)

            if not w or board.is_game_over():
                if self.verbose:
                    print(f'{move_count}. {w}')
                break

            b = await self.black.move(board.fen())
            board.push(b)

            if self.verbose:
                print(f'{move_count}. {w} {b}')

            if not b or board.is_game_over():
                if self.verbose:
                    print(f'{move_count}. {w} {b}')
                break
            move_count += 1
        game = chess.pgn.Game.from_board(board)
        game.headers['White'] = self.white.name
        game.headers['Black'] = self.black.name
        game.headers['Event'] = f'{self.white.name} vs. {self.black.name}'
        game.headers['Date'] = datetime.today().strftime('%Y.%m.%d')
        game.headers['Site'] = 'lukesalamone.com'
        if self.round:
            game.headers['Round'] = str(self.round)
        self.game = game
        return game

    def save(self, path):
        def unique_part():
            return ''.join(random.choices('0123456789abcdef', k=6))

        if Path(path).is_file():
            print(f'Path {path} already exists, saving with different path.')
            original_path = filename
            while Path(path).is_file():
                parts = original_path.split('.')
                path = ''.join(parts[:-1])
                extension = parts[-1]
                path = f'{path}_{unique_part()}.{extension}'

        Path('/'.join(path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            f.write(str(self.game))
