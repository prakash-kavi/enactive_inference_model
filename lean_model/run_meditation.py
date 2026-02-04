"""Main entry point for lean meditation model training and analysis.

Usage:
    python -m lean_model.run_meditation --level expert --timesteps 10000 --seed 42
    python -m lean_model.run_meditation --compare  # Train both expert and novice
"""

import argparse
from pathlib import Path

from .train import train_meditation
from .analysis import (
    plot_training_summary, 
    plot_state_trajectory, 
    compare_expertise,
    print_summary
)


def main():
    parser = argparse.ArgumentParser(
        description='Train and analyze lean meditation model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train expert for 10k steps
  python -m lean_model.run_meditation --level expert --timesteps 10000
  
  # Train novice with custom seed
  python -m lean_model.run_meditation --level novice --timesteps 10000 --seed 123
  
  # Train both and compare
  python -m lean_model.run_meditation --compare --timesteps 10000
  
  # Quick test (1k steps, no plots)
  python -m lean_model.run_meditation --level expert --timesteps 1000 --no-plot
        """
    )
    
    parser.add_argument('--level', type=str, choices=['expert', 'novice'], default='expert',
                       help='Experience level (default: expert)')
    parser.add_argument('--timesteps', type=int, default=10000,
                       help='Number of training steps (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='data/lean_results',
                       help='Output directory for results (default: data/lean_results)')
    parser.add_argument('--plot-dir', type=str, default='plots/lean',
                       help='Output directory for plots (default: plots/lean)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting (only save JSON results)')
    parser.add_argument('--compare', action='store_true',
                       help='Train both expert and novice, then compare')
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.no_plot:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        # Train both phenotypes
        print("\n" + "="*70)
        print("TRAINING EXPERT PHENOTYPE")
        print("="*70)
        expert_results = train_meditation(
            experience_level='expert',
            timesteps=args.timesteps,
            seed=args.seed,
            save_results=True,
            output_dir=args.output_dir
        )
        print_summary(expert_results)
        
        print("\n" + "="*70)
        print("TRAINING NOVICE PHENOTYPE")
        print("="*70)
        novice_results = train_meditation(
            experience_level='novice',
            timesteps=args.timesteps,
            seed=args.seed,
            save_results=True,
            output_dir=args.output_dir
        )
        print_summary(novice_results)
        
        # Comparison plots
        if not args.no_plot:
            print("\nGenerating comparison plots...")
            
            compare_expertise(
                expert_results, 
                novice_results,
                save_path=f"{args.plot_dir}/comparison_seed{args.seed}.png"
            )
            
            plot_training_summary(
                expert_results,
                save_path=f"{args.plot_dir}/training_expert_seed{args.seed}.png"
            )
            
            plot_training_summary(
                novice_results,
                save_path=f"{args.plot_dir}/training_novice_seed{args.seed}.png"
            )
            
            plot_state_trajectory(
                expert_results,
                max_steps=500,
                save_path=f"{args.plot_dir}/trajectory_expert_seed{args.seed}.png"
            )
            
            plot_state_trajectory(
                novice_results,
                max_steps=500,
                save_path=f"{args.plot_dir}/trajectory_novice_seed{args.seed}.png"
            )
            
            print(f"\nAll plots saved to {args.plot_dir}/")
    
    else:
        # Train single level
        print("\n" + "="*70)
        print(f"TRAINING {args.level.upper()} PHENOTYPE")
        print("="*70)
        
        results = train_meditation(
            experience_level=args.level,
            timesteps=args.timesteps,
            seed=args.seed,
            save_results=True,
            output_dir=args.output_dir
        )
        
        print_summary(results)
        
        # Generate plots
        if not args.no_plot:
            print("\nGenerating plots...")
            
            plot_training_summary(
                results,
                save_path=f"{args.plot_dir}/training_{args.level}_seed{args.seed}.png"
            )
            
            plot_state_trajectory(
                results,
                max_steps=500,
                save_path=f"{args.plot_dir}/trajectory_{args.level}_seed{args.seed}.png"
            )
            
            print(f"\nPlots saved to {args.plot_dir}/")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output_dir}/")
    if not args.no_plot:
        print(f"Plots saved to: {args.plot_dir}/")
    print()


if __name__ == '__main__':
    main()
