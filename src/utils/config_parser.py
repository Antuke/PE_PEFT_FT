import argparse
import yaml
import os
from pathlib import Path

def parse_config(parser):
    """
    Parses command line arguments and loads configuration from a YAML file if provided.
    CLI arguments override config file values.
    """
    # Add config argument to the parser
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    
    # Parse known args first to check for config file
    args, remaining_argv = parser.parse_known_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update parser defaults with config values
        parser.set_defaults(**config)
        
    # Parse all args again to allow CLI overrides
    args = parser.parse_args()
    
    return args
