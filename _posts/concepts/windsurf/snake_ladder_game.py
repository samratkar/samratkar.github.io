import random
import time
import os

class SnakeAndLadder:
    def __init__(self):
        # Define snakes (from : to)
        self.snakes = {
            16: 6,
            47: 26,
            49: 11,
            56: 53,
            62: 19,
            64: 60,
            87: 24,
            93: 73,
            95: 75,
            98: 78
        }
        
        # Define ladders (from : to)
        self.ladders = {
            1: 38,
            4: 14,
            9: 31,
            21: 42,
            28: 84,
            36: 44,
            51: 67,
            71: 91,
            80: 100
        }
        
        self.players = {}
        self.player_positions = {}
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_board(self, current_player=None):
        self.clear_screen()
        print("\n🎲 SNAKE 🐍 AND LADDER 🪜 GAME 🎲\n")
        
        # Create the board
        board = []
        num = 100
        for i in range(10):
            row = []
            for j in range(10):
                cell = str(num).rjust(3)
                # Mark player positions
                for player, pos in self.player_positions.items():
                    if pos == num:
                        cell = f" {player} "
                row.append(cell)
                num -= 1
            if i % 2 == 1:  # Reverse alternate rows
                row.reverse()
            board.append(row)
            
        # Print the board
        for row in board:
            print("|", end="")
            for cell in row:
                print(f"{cell}|", end="")
            print("\n" + "-" * 41)
            
        # Print game information
        print("\n🐍 Snakes:", ", ".join([f"{k}→{v}" for k, v in self.snakes.items()]))
        print("🪜 Ladders:", ", ".join([f"{k}→{v}" for k, v in self.ladders.items()]))
        if current_player:
            print(f"\n🎮 Current player: {current_player}")
            
    def roll_dice(self):
        return random.randint(1, 6)
    
    def move_player(self, player, steps):
        current_pos = self.player_positions[player]
        new_pos = current_pos + steps
        
        if new_pos > 100:  # Can't go beyond 100
            return current_pos
            
        # Check for snakes
        if new_pos in self.snakes:
            print(f"🐍 Oops! Snake at {new_pos}! Going down to {self.snakes[new_pos]}")
            time.sleep(1)
            new_pos = self.snakes[new_pos]
            
        # Check for ladders
        elif new_pos in self.ladders:
            print(f"🪜 Yay! Ladder at {new_pos}! Going up to {self.ladders[new_pos]}")
            time.sleep(1)
            new_pos = self.ladders[new_pos]
            
        return new_pos
    
    def play(self):
        # Get number of players
        while True:
            try:
                num_players = int(input("Enter number of players (2-4): "))
                if 2 <= num_players <= 4:
                    break
                print("Please enter a number between 2 and 4")
            except ValueError:
                print("Please enter a valid number")
        
        # Get player names
        for i in range(num_players):
            while True:
                name = input(f"Enter name for Player {i+1}: ").strip()
                if name and name not in self.players:
                    self.players[name] = i+1
                    self.player_positions[name] = 0
                    break
                print("Please enter a unique non-empty name")
        
        # Main game loop
        current_player = list(self.players.keys())[0]
        winner = None
        
        while not winner:
            self.print_board(current_player)
            input(f"\n{current_player}'s turn. Press Enter to roll the dice...")
            
            dice = self.roll_dice()
            print(f"🎲 Rolled a {dice}!")
            time.sleep(1)
            
            # Move player
            old_pos = self.player_positions[current_player]
            new_pos = self.move_player(current_player, dice)
            self.player_positions[current_player] = new_pos
            
            print(f"Moved from {old_pos} to {new_pos}")
            time.sleep(1)
            
            # Check for winner
            if new_pos == 100:
                winner = current_player
                break
                
            # Next player's turn
            current_idx = list(self.players.keys()).index(current_player)
            current_player = list(self.players.keys())[(current_idx + 1) % len(self.players)]
        
        # Game over
        self.print_board()
        print(f"\n🎉 Congratulations {winner}! You've won the game! 🎉")

# Start the game
if __name__ == "__main__":
    game = SnakeAndLadder()
    game.play()
