import sys
import argparse
from src import train, test, demo

def main():
    parser = argparse.ArgumentParser(description="Framework Entry Point")
    parser.add_argument('command', choices=['train', 'test', 'demo'], help="Command to execute")
    
    # Parse only the command, leave the rest for the scripts
    args, remaining_args = parser.parse_known_args()
    
    # Update sys.argv to remove the command (e.g., 'train') so the scripts don't see it
    sys.argv = [sys.argv[0]] + remaining_args
    
    if args.command == 'train':
        train.main()
    elif args.command == 'test':
        test.main()
    elif args.command == 'demo':
        demo.main()

if __name__ == "__main__":
    main()
