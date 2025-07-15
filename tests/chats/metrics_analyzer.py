"""
Metrics Analyzer for Chat API Test Results
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")


@dataclass
class TestSummary:
    """Summary statistics for test results"""
    total_tests: int
    successful_tests: int
    failed_tests: int
    success_rate: float
    avg_response_time: float
    median_response_time: float
    avg_confidence: float
    median_confidence: float
    min_response_time: float
    max_response_time: float
    tests_with_refinement: int
    tests_without_refinement: int


class MetricsAnalyzer:
    """Analyze and report on chat API test metrics"""
    
    def __init__(self, metrics_file: str = None):
        self.metrics_file = metrics_file
        self.metrics_data = []
        self.df = None
        
        if metrics_file:
            self.load_metrics(metrics_file)
    
    def load_metrics(self, metrics_file: str):
        """Load metrics from JSON file"""
        file_path = Path(metrics_file)
        if not file_path.exists():
            # Try looking in the results directory
            results_dir = Path("tests/chats/results")
            file_path = results_dir / metrics_file
            
        if not file_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.metrics_data = json.load(f)
            
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.metrics_data)
        
        # Clean up data types
        self.df['response_time'] = pd.to_numeric(self.df['response_time'])
        self.df['confidence'] = pd.to_numeric(self.df['confidence'])
        self.df['success'] = self.df['success'].astype(bool)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Extract refinement info
        self.df['enable_refinement'] = self.df['request_data'].apply(
            lambda x: x.get('enable_refinement', True)
        )
        
        print(f"Loaded {len(self.metrics_data)} test metrics from {file_path}")
    
    def get_summary(self) -> TestSummary:
        """Get summary statistics"""
        if self.df is None:
            raise ValueError("No metrics loaded")
        
        successful_tests = self.df['success'].sum()
        total_tests = len(self.df)
        
        return TestSummary(
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=total_tests - successful_tests,
            success_rate=successful_tests / total_tests * 100,
            avg_response_time=self.df['response_time'].mean(),
            median_response_time=self.df['response_time'].median(),
            avg_confidence=self.df[self.df['success']]['confidence'].mean(),
            median_confidence=self.df[self.df['success']]['confidence'].median(),
            min_response_time=self.df['response_time'].min(),
            max_response_time=self.df['response_time'].max(),
            tests_with_refinement=self.df['enable_refinement'].sum(),
            tests_without_refinement=(~self.df['enable_refinement']).sum()
        )
    
    def analyze_by_scenario(self) -> pd.DataFrame:
        """Analyze metrics by test scenario"""
        if self.df is None:
            raise ValueError("No metrics loaded")
        
        # Extract scenario names from test names or request messages
        scenario_analysis = []
        
        for _, row in self.df.iterrows():
            message = row['request_data']['message']
            scenario_name = self._extract_scenario_name(message)
            
            scenario_analysis.append({
                'scenario': scenario_name,
                'message': message,
                'success': row['success'],
                'response_time': row['response_time'],
                'confidence': row['confidence'],
                'enable_refinement': row['enable_refinement'],
                'actions_count': len(row['actions_taken']),
                'next_steps_count': len(row['next_steps'])
            })
        
        scenario_df = pd.DataFrame(scenario_analysis)
        
        # Group by scenario and calculate statistics
        grouped = scenario_df.groupby('scenario').agg({
            'success': ['count', 'sum', 'mean'],
            'response_time': ['mean', 'median', 'std'],
            'confidence': ['mean', 'median', 'std'],
            'actions_count': ['mean', 'median'],
            'next_steps_count': ['mean', 'median']
        }).round(2)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
        return grouped.reset_index()
    
    def _extract_scenario_name(self, message: str) -> str:
        """Extract scenario name from message"""
        message_lower = message.lower()
        
        if "london" in message_lower:
            return "london_trip"
        elif "paris" in message_lower:
            return "paris_trip"
        elif "tokyo" in message_lower:
            return "tokyo_trip"
        elif "singapore" in message_lower:
            return "singapore_trip"
        elif "europe" in message_lower:
            return "europe_backpacking"
        elif "dubai" in message_lower:
            return "dubai_luxury"
        elif "barcelona" in message_lower:
            return "barcelona_solo"
        elif "ski" in message_lower or "alps" in message_lower:
            return "ski_trip"
        else:
            return "other"
    
    def compare_refinement_impact(self) -> Dict:
        """Compare performance with and without refinement"""
        if self.df is None:
            raise ValueError("No metrics loaded")
        
        with_refinement = self.df[self.df['enable_refinement'] == True]
        without_refinement = self.df[self.df['enable_refinement'] == False]
        
        comparison = {}
        
        # Only include data for groups that exist
        if len(with_refinement) > 0:
            successful_with = with_refinement[with_refinement['success']]
            comparison['with_refinement'] = {
                'count': len(with_refinement),
                'success_rate': with_refinement['success'].mean() * 100,
                'avg_response_time': with_refinement['response_time'].mean(),
                'avg_confidence': successful_with['confidence'].mean() if len(successful_with) > 0 else 0.0,
                'median_response_time': with_refinement['response_time'].median()
            }
        
        if len(without_refinement) > 0:
            successful_without = without_refinement[without_refinement['success']]
            comparison['without_refinement'] = {
                'count': len(without_refinement),
                'success_rate': without_refinement['success'].mean() * 100,
                'avg_response_time': without_refinement['response_time'].mean(),
                'avg_confidence': successful_without['confidence'].mean() if len(successful_without) > 0 else 0.0,
                'median_response_time': without_refinement['response_time'].median()
            }
        
        return comparison
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive analysis report"""
        if self.df is None:
            raise ValueError("No metrics loaded")
        
        summary = self.get_summary()
        scenario_analysis = self.analyze_by_scenario()
        refinement_comparison = self.compare_refinement_impact()
        
        report = []
        report.append("=" * 60)
        report.append("CHAT API TEST METRICS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Metrics file: {self.metrics_file}")
        report.append("")
        
        # Overall Summary
        report.append("OVERALL SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Tests: {summary.total_tests}")
        report.append(f"Successful Tests: {summary.successful_tests}")
        report.append(f"Failed Tests: {summary.failed_tests}")
        report.append(f"Success Rate: {summary.success_rate:.1f}%")
        report.append(f"Average Response Time: {summary.avg_response_time:.2f}s")
        report.append(f"Median Response Time: {summary.median_response_time:.2f}s")
        report.append(f"Average Confidence: {summary.avg_confidence:.2f}")
        report.append(f"Response Time Range: {summary.min_response_time:.2f}s - {summary.max_response_time:.2f}s")
        report.append("")
        
        # Refinement Impact
        report.append("REFINEMENT IMPACT ANALYSIS")
        report.append("-" * 30)
        
        if 'with_refinement' in refinement_comparison:
            with_ref = refinement_comparison['with_refinement']
            report.append(f"With Refinement ({with_ref['count']} tests):")
            report.append(f"  Success Rate: {with_ref['success_rate']:.1f}%")
            report.append(f"  Avg Response Time: {with_ref['avg_response_time']:.2f}s")
            report.append(f"  Avg Confidence: {with_ref['avg_confidence']:.2f}")
            report.append("")
        
        if 'without_refinement' in refinement_comparison:
            without_ref = refinement_comparison['without_refinement']
            report.append(f"Without Refinement ({without_ref['count']} tests):")
            report.append(f"  Success Rate: {without_ref['success_rate']:.1f}%")
            report.append(f"  Avg Response Time: {without_ref['avg_response_time']:.2f}s")
            report.append(f"  Avg Confidence: {without_ref['avg_confidence']:.2f}")
            report.append("")
        
        # Performance insights (only if both groups exist)
        if 'with_refinement' in refinement_comparison and 'without_refinement' in refinement_comparison:
            with_ref = refinement_comparison['with_refinement']
            without_ref = refinement_comparison['without_refinement']
            time_diff = with_ref['avg_response_time'] - without_ref['avg_response_time']
            confidence_diff = with_ref['avg_confidence'] - without_ref['avg_confidence']
            
            report.append("Key Insights:")
            report.append(f"  Response Time Impact: {time_diff:+.2f}s ({'slower' if time_diff > 0 else 'faster'} with refinement)")
            report.append(f"  Confidence Impact: {confidence_diff:+.2f} ({'higher' if confidence_diff > 0 else 'lower'} with refinement)")
            report.append("")
        elif 'with_refinement' in refinement_comparison:
            report.append("Key Insights:")
            report.append("  Only refinement-enabled tests available")
            report.append("")
        elif 'without_refinement' in refinement_comparison:
            report.append("Key Insights:")
            report.append("  Only refinement-disabled tests available")
            report.append("")
        
        # Scenario Analysis
        report.append("SCENARIO ANALYSIS")
        report.append("-" * 20)
        for _, row in scenario_analysis.iterrows():
            scenario = row['scenario']
            success_rate = row['success_mean'] * 100
            avg_time = row['response_time_mean']
            avg_confidence = row['confidence_mean']
            
            report.append(f"{scenario}:")
            report.append(f"  Tests: {row['success_count']}")
            report.append(f"  Success Rate: {success_rate:.1f}%")
            report.append(f"  Avg Response Time: {avg_time:.2f}s")
            report.append(f"  Avg Confidence: {avg_confidence:.2f}")
            report.append("")
        
        # Performance Benchmarks
        report.append("PERFORMANCE BENCHMARKS")
        report.append("-" * 25)
        
        # Response time categories (adjusted for ~1 minute normal response time)
        fast_tests = len(self.df[self.df['response_time'] < 45.0])
        medium_tests = len(self.df[(self.df['response_time'] >= 45.0) & (self.df['response_time'] < 75.0)])
        slow_tests = len(self.df[self.df['response_time'] >= 75.0])
        
        report.append(f"Fast responses (<45s): {fast_tests} ({fast_tests/summary.total_tests*100:.1f}%)")
        report.append(f"Medium responses (45-75s): {medium_tests} ({medium_tests/summary.total_tests*100:.1f}%)")
        report.append(f"Slow responses (>75s): {slow_tests} ({slow_tests/summary.total_tests*100:.1f}%)")
        report.append("")
        
        # Confidence categories
        high_conf = len(self.df[self.df['confidence'] >= 0.8])
        medium_conf = len(self.df[(self.df['confidence'] >= 0.6) & (self.df['confidence'] < 0.8)])
        low_conf = len(self.df[self.df['confidence'] < 0.6])
        
        report.append(f"High confidence (>=0.8): {high_conf} ({high_conf/summary.total_tests*100:.1f}%)")
        report.append(f"Medium confidence (0.6-0.8): {medium_conf} ({medium_conf/summary.total_tests*100:.1f}%)")
        report.append(f"Low confidence (<0.6): {low_conf} ({low_conf/summary.total_tests*100:.1f}%)")
        report.append("")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if output_file:
            output_path = Path("tests/chats/results") / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"Report saved to: {output_path}")
        
        return report_text
    
    def create_visualizations(self, output_dir: str = "tests/chats/results"):
        """Create visualizations of the metrics"""
        if self.df is None:
            raise ValueError("No metrics loaded")
        
        if not PLOTTING_AVAILABLE:
            print("Skipping visualizations - matplotlib/seaborn not available")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up the plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            # Fall back to seaborn if seaborn-v0_8 not available
            try:
                plt.style.use('seaborn')
            except OSError:
                # Use default style if seaborn not available
                plt.style.use('default')
        
        # 1. Response Time Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['response_time'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Response Times')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'response_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Success Rate by Refinement
        refinement_stats = self.df.groupby('enable_refinement')['success'].agg(['count', 'sum', 'mean'])
        
        plt.figure(figsize=(8, 6))
        
        # Check which refinement types are available
        has_false = False in refinement_stats.index
        has_true = True in refinement_stats.index
        
        categories = []
        success_rates = []
        colors = []
        
        if has_false:
            categories.append('Without Refinement')
            success_rates.append(refinement_stats.loc[False, 'mean'] * 100)
            colors.append('lightcoral')
        
        if has_true:
            categories.append('With Refinement')
            success_rates.append(refinement_stats.loc[True, 'mean'] * 100)
            colors.append('lightgreen')
        
        if len(categories) == 0:
            plt.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Success Rate: With vs Without Refinement')
        else:
            bars = plt.bar(categories, success_rates, color=colors, alpha=0.7)
            plt.ylabel('Success Rate (%)')
            plt.title('Success Rate: With vs Without Refinement')
            plt.ylim(0, 100)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.savefig(output_path / 'success_rate_by_refinement.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confidence vs Response Time Scatter Plot
        plt.figure(figsize=(10, 6))
        successful_tests = self.df[self.df['success'] == True]
        
        if len(successful_tests) > 0:
            # Check if we have both refinement types
            refinement_values = successful_tests['enable_refinement'].unique()
            
            if len(refinement_values) > 1:
                # Multiple refinement types - use color coding
                scatter = plt.scatter(successful_tests['response_time'], successful_tests['confidence'], 
                                    c=successful_tests['enable_refinement'], cmap='RdYlBu', alpha=0.7)
                plt.colorbar(scatter, label='Refinement Enabled')
            else:
                # Single refinement type - use single color
                color = 'lightgreen' if refinement_values[0] else 'lightcoral'
                label = f"Refinement {'Enabled' if refinement_values[0] else 'Disabled'}"
                plt.scatter(successful_tests['response_time'], successful_tests['confidence'], 
                           c=color, alpha=0.7, label=label)
                plt.legend()
            
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Confidence Score')
            plt.title('Confidence Score vs Response Time')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No successful tests to plot', ha='center', va='center', transform=plt.gca().transAxes)
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Confidence Score')
            plt.title('Confidence Score vs Response Time')
        
        plt.savefig(output_path / 'confidence_vs_response_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_path}")


def analyze_latest_metrics():
    """Analyze the most recent metrics file"""
    results_dir = Path("tests/chats/results")
    
    if not results_dir.exists():
        print("No results directory found")
        return
    
    # Find the most recent metrics file
    metrics_files = list(results_dir.glob("*.json"))
    if not metrics_files:
        print("No metrics files found")
        return
    
    latest_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
    print(f"Analyzing latest metrics file: {latest_file}")
    
    analyzer = MetricsAnalyzer(str(latest_file))
    
    # Generate report
    report = analyzer.generate_report("analysis_report.txt")
    print(report)
    
    # Create visualizations
    analyzer.create_visualizations()


if __name__ == "__main__":
    analyze_latest_metrics() 