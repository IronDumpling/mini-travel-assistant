"""
Enhanced Metrics Analyzer for Advanced Chat API Test Results
Supports analysis of refinement loops and two test types
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn/numpy not available. Visualizations will be skipped.")


@dataclass
class RefinementAnalysis:
    """Analysis of refinement loop performance"""
    avg_loops_per_test: float
    max_loops_observed: int
    min_loops_observed: int
    confidence_improvement: Dict[str, float]  # improvement per loop
    time_cost_per_loop: Dict[str, float]
    success_rate_by_loops: Dict[int, float]


@dataclass
class TestTypeComparison:
    """Comparison between single-session and multi-session tests"""
    single_session_stats: Dict[str, Any]
    multi_session_stats: Dict[str, Any]
    performance_differences: Dict[str, float]


class MetricsAnalyzer:
    """Analyzer for the test framework"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.single_session_data = []
        self.multi_session_data = []
        self.summary_data = {}
        
        self.load_results()
    
    def load_results(self):
        """Load all test results from the directory"""
        
        # Load single session results
        single_file = self.results_dir / "single_session_tests.json"
        if single_file.exists():
            with open(single_file, 'r', encoding='utf-8') as f:
                self.single_session_data = json.load(f)
        
        # Load multi session results
        multi_file = self.results_dir / "multi_session_tests.json"
        if multi_file.exists():
            with open(multi_file, 'r', encoding='utf-8') as f:
                self.multi_session_data = json.load(f)
        
        # Load summary
        summary_file = self.results_dir / "test_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                self.summary_data = json.load(f)
        
        print(f"Loaded results from: {self.results_dir}")
        print(f"Single-session tests: {len(self.single_session_data)}")
        print(f"Multi-session suites: {len(self.multi_session_data)}")
    
    def analyze_refinement_performance(self) -> RefinementAnalysis:
        """Analyze refinement loop performance across all tests"""
        
        all_metrics = []
        
        # Collect all metrics from both test types
        all_metrics.extend(self.single_session_data)
        for suite in self.multi_session_data:
            all_metrics.extend(suite["metrics"])
        
        # Filter tests with refinement enabled
        refinement_tests = [m for m in all_metrics if m["refinement_enabled"]]
        
        if not refinement_tests:
            return RefinementAnalysis(0, 0, 0, {}, {}, {})
        
        # Basic loop statistics
        loop_counts = [m["total_loops"] for m in refinement_tests]
        avg_loops = sum(loop_counts) / len(loop_counts)
        max_loops = max(loop_counts)
        min_loops = min(loop_counts)
        
        # Confidence improvement analysis
        confidence_improvements = {}
        time_costs = {}
        
        for metric in refinement_tests:
            loops = metric["refinement_loops"]
            if len(loops) > 1:
                initial_confidence = loops[0]["confidence"]
                final_confidence = loops[-1]["confidence"]
                improvement = final_confidence - initial_confidence
                confidence_improvements[metric["test_id"]] = improvement
                
                # Time cost per loop
                total_loop_time = sum(loop["response_time"] for loop in loops)
                time_costs[metric["test_id"]] = total_loop_time / len(loops)
        
        # Success rate by number of loops
        success_by_loops = {}
        for loop_count in set(loop_counts):
            tests_with_count = [m for m in refinement_tests if m["total_loops"] == loop_count]
            successful = sum(1 for m in tests_with_count if m["final_success"])
            success_by_loops[loop_count] = successful / len(tests_with_count) * 100
        
        return RefinementAnalysis(
            avg_loops_per_test=avg_loops,
            max_loops_observed=max_loops,
            min_loops_observed=min_loops,
            confidence_improvement=confidence_improvements,
            time_cost_per_loop=time_costs,
            success_rate_by_loops=success_by_loops
        )
    
    def compare_test_types(self) -> TestTypeComparison:
        """Compare single-session vs multi-session test performance"""
        
        # Single-session statistics
        single_metrics = self.single_session_data
        single_stats = {
            "total_tests": len(single_metrics),
            "success_rate": sum(1 for m in single_metrics if m["final_success"]) / len(single_metrics) * 100 if single_metrics else 0,
            "avg_response_time": sum(m["total_response_time"] for m in single_metrics) / len(single_metrics) if single_metrics else 0,
            "avg_confidence": sum(m["final_confidence"] for m in single_metrics if m["final_success"]) / max(sum(1 for m in single_metrics if m["final_success"]), 1),
            "avg_refinement_loops": sum(m["total_loops"] for m in single_metrics) / len(single_metrics) if single_metrics else 0
        }
        
        # Multi-session statistics (flattened)
        multi_metrics = []
        for suite in self.multi_session_data:
            multi_metrics.extend(suite["metrics"])
        
        multi_stats = {
            "total_tests": len(multi_metrics),
            "success_rate": sum(1 for m in multi_metrics if m["final_success"]) / len(multi_metrics) * 100 if multi_metrics else 0,
            "avg_response_time": sum(m["total_response_time"] for m in multi_metrics) / len(multi_metrics) if multi_metrics else 0,
            "avg_confidence": sum(m["final_confidence"] for m in multi_metrics if m["final_success"]) / max(sum(1 for m in multi_metrics if m["final_success"]), 1),
            "avg_refinement_loops": sum(m["total_loops"] for m in multi_metrics) / len(multi_metrics) if multi_metrics else 0,
            "total_sessions": len(self.multi_session_data),
            "avg_queries_per_session": len(multi_metrics) / len(self.multi_session_data) if self.multi_session_data else 0
        }
        
        # Calculate performance differences
        differences = {}
        if single_stats["total_tests"] > 0 and multi_stats["total_tests"] > 0:
            differences = {
                "success_rate_diff": multi_stats["success_rate"] - single_stats["success_rate"],
                "response_time_diff": multi_stats["avg_response_time"] - single_stats["avg_response_time"],
                "confidence_diff": multi_stats["avg_confidence"] - single_stats["avg_confidence"],
                "refinement_loops_diff": multi_stats["avg_refinement_loops"] - single_stats["avg_refinement_loops"]
            }
        
        return TestTypeComparison(
            single_session_stats=single_stats,
            multi_session_stats=multi_stats,
            performance_differences=differences
        )
    
    def analyze_session_progression(self) -> Dict[str, Any]:
        """Analyze how performance changes across queries within multi-session tests"""
        
        session_progressions = {}
        
        for suite in self.multi_session_data:
            session_id = suite["session_id"]
            metrics = suite["metrics"]
            
            progression = {
                "query_count": len(metrics),
                "success_progression": [m["final_success"] for m in metrics],
                "confidence_progression": [m["final_confidence"] for m in metrics],
                "response_time_progression": [m["total_response_time"] for m in metrics],
                "refinement_loops_progression": [m["total_loops"] for m in metrics]
            }
            
            # Calculate trends
            if len(metrics) > 1:
                confidences = [m["final_confidence"] for m in metrics if m["final_success"]]
                response_times = [m["total_response_time"] for m in metrics]
                
                progression["confidence_trend"] = "improving" if len(confidences) > 1 and confidences[-1] > confidences[0] else "stable"
                progression["response_time_trend"] = "improving" if response_times[-1] < response_times[0] else "stable"
            
            session_progressions[session_id] = progression
        
        return session_progressions
    
    def generate_detailed_report(self) -> str:
        """Generate comprehensive analysis report"""
        
        refinement_analysis = self.analyze_refinement_performance()
        test_comparison = self.compare_test_types()
        session_progression = self.analyze_session_progression()
        
        report = []
        report.append("=" * 80)
        report.append("ENHANCED CHAT API TEST ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Results Directory: {self.results_dir}")
        if "test_run_id" in self.summary_data:
            report.append(f"Test Run ID: {self.summary_data['test_run_id']}")
        report.append("")
        
        # Overall Summary
        report.append("📊 OVERALL SUMMARY")
        report.append("-" * 40)
        single_stats = test_comparison.single_session_stats
        multi_stats = test_comparison.multi_session_stats
        
        report.append(f"Single-Session Tests: {single_stats['total_tests']}")
        report.append(f"Multi-Session Tests: {multi_stats['total_tests']} queries across {multi_stats.get('total_sessions', 0)} sessions")
        report.append(f"Total Tests: {single_stats['total_tests'] + multi_stats['total_tests']}")
        report.append("")
        
        # Test Type Comparison
        report.append("🔄 TEST TYPE COMPARISON")
        report.append("-" * 40)
        
        report.append("Single-Session Tests:")
        report.append(f"  Success Rate: {single_stats['success_rate']:.1f}%")
        report.append(f"  Avg Response Time: {single_stats['avg_response_time']:.2f}s")
        report.append(f"  Avg Confidence: {single_stats['avg_confidence']:.2f}")
        report.append(f"  Avg Refinement Loops: {single_stats['avg_refinement_loops']:.1f}")
        report.append("")
        
        report.append("Multi-Session Tests:")
        report.append(f"  Success Rate: {multi_stats['success_rate']:.1f}%")
        report.append(f"  Avg Response Time: {multi_stats['avg_response_time']:.2f}s")
        report.append(f"  Avg Confidence: {multi_stats['avg_confidence']:.2f}")
        report.append(f"  Avg Refinement Loops: {multi_stats['avg_refinement_loops']:.1f}")
        report.append(f"  Avg Queries per Session: {multi_stats.get('avg_queries_per_session', 0):.1f}")
        report.append("")
        
        # Performance differences
        if test_comparison.performance_differences:
            diffs = test_comparison.performance_differences
            report.append("Performance Differences (Multi-Session vs Single-Session):")
            report.append(f"  Success Rate: {diffs['success_rate_diff']:+.1f}%")
            report.append(f"  Response Time: {diffs['response_time_diff']:+.2f}s")
            report.append(f"  Confidence: {diffs['confidence_diff']:+.2f}")
            report.append(f"  Refinement Loops: {diffs['refinement_loops_diff']:+.1f}")
            report.append("")
        
        # Refinement Analysis
        report.append("🔧 REFINEMENT LOOP ANALYSIS")
        report.append("-" * 40)
        report.append(f"Average Loops per Test: {refinement_analysis.avg_loops_per_test:.1f}")
        report.append(f"Loop Range: {refinement_analysis.min_loops_observed} - {refinement_analysis.max_loops_observed}")
        
        if refinement_analysis.success_rate_by_loops:
            report.append("\nSuccess Rate by Loop Count:")
            for loops, rate in sorted(refinement_analysis.success_rate_by_loops.items()):
                report.append(f"  {loops} loops: {rate:.1f}% success")
        
        if refinement_analysis.confidence_improvement:
            avg_improvement = sum(refinement_analysis.confidence_improvement.values()) / len(refinement_analysis.confidence_improvement)
            report.append(f"\nAverage Confidence Improvement: {avg_improvement:+.3f}")
        
        if refinement_analysis.time_cost_per_loop:
            avg_time_cost = sum(refinement_analysis.time_cost_per_loop.values()) / len(refinement_analysis.time_cost_per_loop)
            report.append(f"Average Time Cost per Loop: {avg_time_cost:.2f}s")
        
        report.append("")
        
        # Session Progression Analysis
        if session_progression:
            report.append("📈 SESSION PROGRESSION ANALYSIS")
            report.append("-" * 40)
            
            improving_confidence = sum(1 for p in session_progression.values() if p.get("confidence_trend") == "improving")
            improving_response_time = sum(1 for p in session_progression.values() if p.get("response_time_trend") == "improving")
            
            report.append(f"Sessions with Improving Confidence: {improving_confidence}/{len(session_progression)}")
            report.append(f"Sessions with Improving Response Time: {improving_response_time}/{len(session_progression)}")
            
            # Show sample session progression
            for i, (session_id, progression) in enumerate(list(session_progression.items())[:3]):
                report.append(f"\nSample Session {i+1} ({session_id[:8]}...):")
                report.append(f"  Queries: {progression['query_count']}")
                success_rate = sum(progression['success_progression']) / len(progression['success_progression']) * 100
                report.append(f"  Success Rate: {success_rate:.1f}%")
                if progression['confidence_progression']:
                    avg_confidence = sum(c for c, s in zip(progression['confidence_progression'], progression['success_progression']) if s) / max(sum(progression['success_progression']), 1)
                    report.append(f"  Avg Confidence: {avg_confidence:.2f}")
        
        report.append("")
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        report_file = self.results_dir / "enhanced_analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        print(f"Enhanced analysis report saved to: {report_file}")
        return report_text
    
    def create_visualizations(self):
        """Create enhanced visualizations for the new metrics"""
        
        if not PLOTTING_AVAILABLE:
            print("Skipping visualizations - matplotlib/seaborn not available")
            return
        
        # Set style
        plt.style.use('default')
        if 'seaborn' in plt.style.available:
            plt.style.use('seaborn-v0_8-whitegrid')
        
        # 📊 原有的图表
        # 1. Refinement Loop Distribution (修复后)
        self._plot_refinement_distribution()
        
        # 2. Test Type Performance Comparison
        self._plot_test_type_comparison()
        
        # 3. Confidence vs Loops Scatter
        self._plot_confidence_vs_loops()
        
        # 4. Session Progression Analysis
        self._plot_session_progression()
        
        # 5. Response Time vs Refinement Loops
        self._plot_response_time_vs_loops()
        
        # 📊 新增的图表
        # 6. 响应时间频率分布图
        self._plot_response_time_distribution()
        
        # 7. 响应评估结果分数 vs refinement循环次数分布图
        self._plot_confidence_vs_refinement_scatter()
        
        print(f"Enhanced visualizations saved to: {self.results_dir}")

    # ✅ 更新方法名称以保持一致性
    def create_enhanced_visualizations(self):
        """Create enhanced visualizations for the new metrics"""
        return self.create_visualizations()
    
    def _plot_refinement_distribution(self):
        """Plot distribution of refinement loops"""
        all_metrics = []
        all_metrics.extend(self.single_session_data)
        for suite in self.multi_session_data:
            all_metrics.extend(suite["metrics"])
        
        refinement_tests = [m for m in all_metrics if m["refinement_enabled"]]
        loop_counts = [m["total_loops"] for m in refinement_tests]
        
        plt.figure(figsize=(10, 6))
        
        # ✅ 修复：检查是否有refinement数据
        if not loop_counts:
            # 如果没有refinement数据，显示说明性图表
            plt.text(0.5, 0.5, 'No Refinement Data Available\n(All tests ran without refinement)', 
                    ha='center', va='center', transform=plt.gca().transAxes, 
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title('Distribution of Refinement Loops - No Data')
        else:
            max_loops = max(loop_counts)
            plt.hist(loop_counts, bins=range(max_loops + 2), alpha=0.7, edgecolor='black')
            plt.xlabel('Number of Refinement Loops')
            plt.ylabel('Frequency')
            plt.title('Distribution of Refinement Loops')
        
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'refinement_loop_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_response_time_distribution(self):
        """📊 新增：响应时间频率分布图"""
        all_metrics = []
        all_metrics.extend(self.single_session_data)
        for suite in self.multi_session_data:
            all_metrics.extend(suite["metrics"])
        
        # 获取所有响应时间
        response_times = [m["total_response_time"] for m in all_metrics if m.get("total_response_time")]
        
        plt.figure(figsize=(12, 6))
        
        if not response_times:
            plt.text(0.5, 0.5, 'No Response Time Data Available', 
                    ha='center', va='center', transform=plt.gca().transAxes, 
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title('Response Time Distribution - No Data')
        else:
            # 创建直方图
            plt.hist(response_times, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Frequency')
            plt.title('Response Time Frequency Distribution')
            
            # 添加统计信息
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            plt.axvline(avg_time, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_time:.1f}s')
            plt.axvline(max_time, color='orange', linestyle='--', linewidth=1, label=f'Max: {max_time:.1f}s')
            plt.axvline(min_time, color='green', linestyle='--', linewidth=1, label=f'Min: {min_time:.1f}s')
            
            plt.legend()
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'response_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_vs_refinement_scatter(self):
        """📊 新增：响应评估结果分数 vs refinement循环次数分布图"""
        all_metrics = []
        all_metrics.extend(self.single_session_data)
        for suite in self.multi_session_data:
            all_metrics.extend(suite["metrics"])
        
        # 收集所有数据点
        refinement_loops = []
        confidence_scores = []
        response_times = []
        test_types = []
        
        for m in all_metrics:
            loops = m.get("total_loops", 0)
            confidence = m.get("final_confidence", 0)
            response_time = m.get("total_response_time", 0)
            test_type = "Multi-Session" if any(m in suite["metrics"] for suite in self.multi_session_data) else "Single-Session"
            
            refinement_loops.append(loops)
            confidence_scores.append(confidence)
            response_times.append(response_time)
            test_types.append(test_type)
        
        plt.figure(figsize=(14, 10))
        
        if not refinement_loops:
            plt.text(0.5, 0.5, 'No Test Data Available', 
                    ha='center', va='center', transform=plt.gca().transAxes, 
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title('Confidence vs Refinement Loops - No Data')
        else:
            # 创建子图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 置信度 vs 循环次数散点图
            single_mask = [t == "Single-Session" for t in test_types]
            multi_mask = [t == "Multi-Session" for t in test_types]
            
            if any(single_mask):
                ax1.scatter([refinement_loops[i] for i, x in enumerate(single_mask) if x], 
                           [confidence_scores[i] for i, x in enumerate(single_mask) if x], 
                           alpha=0.6, label='Single-Session', color='blue', s=60)
            if any(multi_mask):
                ax1.scatter([refinement_loops[i] for i, x in enumerate(multi_mask) if x], 
                           [confidence_scores[i] for i, x in enumerate(multi_mask) if x], 
                           alpha=0.6, label='Multi-Session', color='orange', s=60)
            
            ax1.set_xlabel('Number of Refinement Loops')
            ax1.set_ylabel('Final Confidence Score')
            ax1.set_title('Confidence Score vs Refinement Loops')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim(0, 1)
            
            # 2. 响应时间 vs 循环次数
            if any(single_mask):
                ax2.scatter([refinement_loops[i] for i, x in enumerate(single_mask) if x], 
                           [response_times[i] for i, x in enumerate(single_mask) if x], 
                           alpha=0.6, label='Single-Session', color='blue', s=60)
            if any(multi_mask):
                ax2.scatter([refinement_loops[i] for i, x in enumerate(multi_mask) if x], 
                           [response_times[i] for i, x in enumerate(multi_mask) if x], 
                           alpha=0.6, label='Multi-Session', color='orange', s=60)
            
            ax2.set_xlabel('Number of Refinement Loops')
            ax2.set_ylabel('Response Time (seconds)')
            ax2.set_title('Response Time vs Refinement Loops')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 3. 置信度分布箱线图
            single_confidences = [confidence_scores[i] for i, x in enumerate(single_mask) if x]
            multi_confidences = [confidence_scores[i] for i, x in enumerate(multi_mask) if x]
            
            box_data = []
            box_labels = []
            if single_confidences:
                box_data.append(single_confidences)
                box_labels.append('Single-Session')
            if multi_confidences:
                box_data.append(multi_confidences)
                box_labels.append('Multi-Session')
            
            if box_data:
                ax3.boxplot(box_data, labels=box_labels)
                ax3.set_ylabel('Confidence Score')
                ax3.set_title('Confidence Score Distribution by Test Type')
                ax3.grid(True, alpha=0.3)
            
            # 4. 响应时间分布箱线图
            single_times = [response_times[i] for i, x in enumerate(single_mask) if x]
            multi_times = [response_times[i] for i, x in enumerate(multi_mask) if x]
            
            time_box_data = []
            time_box_labels = []
            if single_times:
                time_box_data.append(single_times)
                time_box_labels.append('Single-Session')
            if multi_times:
                time_box_data.append(multi_times)
                time_box_labels.append('Multi-Session')
            
            if time_box_data:
                ax4.boxplot(time_box_data, labels=time_box_labels)
                ax4.set_ylabel('Response Time (seconds)')
                ax4.set_title('Response Time Distribution by Test Type')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        plt.savefig(self.results_dir / 'confidence_vs_refinement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_test_type_comparison(self):
        """Plot comparison between test types"""
        comparison = self.compare_test_types()
        
        metrics = ['success_rate', 'avg_response_time', 'avg_confidence', 'avg_refinement_loops']
        single_values = [comparison.single_session_stats[m] for m in metrics]
        multi_values = [comparison.multi_session_stats[m] for m in metrics]
        
        x = range(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 8))
        plt.bar([i - width/2 for i in x], single_values, width, label='Single-Session', alpha=0.7)
        plt.bar([i + width/2 for i in x], multi_values, width, label='Multi-Session', alpha=0.7)
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Test Type Performance Comparison')
        plt.xticks(x, ['Success Rate (%)', 'Avg Response Time (s)', 'Avg Confidence', 'Avg Refinement Loops'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'test_type_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_vs_loops(self):
        """Plot confidence vs number of refinement loops"""
        all_metrics = []
        all_metrics.extend(self.single_session_data)
        for suite in self.multi_session_data:
            all_metrics.extend(suite["metrics"])
        
        refinement_tests = [m for m in all_metrics if m["refinement_enabled"] and m["final_success"]]
        
        loops = [m["total_loops"] for m in refinement_tests]
        confidences = [m["final_confidence"] for m in refinement_tests]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(loops, confidences, alpha=0.6)
        plt.xlabel('Number of Refinement Loops')
        plt.ylabel('Final Confidence Score')
        plt.title('Confidence Score vs Refinement Loops')
        plt.grid(True, alpha=0.3)
        
        # Add trend line if enough data
        if len(loops) > 5:
            z = np.polyfit(loops, confidences, 1)
            p = np.poly1d(z)
            plt.plot(sorted(set(loops)), p(sorted(set(loops))), "r--", alpha=0.8)
        
        plt.savefig(self.results_dir / 'confidence_vs_loops.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_session_progression(self):
        """Plot session progression for multi-session tests"""
        if not self.multi_session_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample a few sessions for detailed progression plots
        sample_sessions = self.multi_session_data[:4]  # First 4 sessions
        
        for i, suite in enumerate(sample_sessions):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            metrics = suite["metrics"]
            query_indices = range(len(metrics))
            confidences = [m["final_confidence"] if m["final_success"] else 0 for m in metrics]
            
            ax.plot(query_indices, confidences, 'o-', alpha=0.7)
            ax.set_xlabel('Query Index')
            ax.set_ylabel('Confidence Score')
            ax.set_title(f'Session: {suite["session_title"][:30]}...')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'session_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_response_time_vs_loops(self):
        """Plot response time vs refinement loops"""
        all_metrics = []
        all_metrics.extend(self.single_session_data)
        for suite in self.multi_session_data:
            all_metrics.extend(suite["metrics"])
        
        refinement_tests = [m for m in all_metrics if m["refinement_enabled"]]
        
        loops = [m["total_loops"] for m in refinement_tests]
        response_times = [m["total_response_time"] for m in refinement_tests]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(loops, response_times, alpha=0.6)
        plt.xlabel('Number of Refinement Loops')
        plt.ylabel('Total Response Time (seconds)')
        plt.title('Response Time vs Refinement Loops')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'response_time_vs_loops.png', dpi=300, bbox_inches='tight')
        plt.close()


def analyze_latest_results():
    """Analyze the most recent test results"""
    results_base = Path("tests/chats/results")
    
    if not results_base.exists():
        print("No results directory found")
        return
    
    # Find the most recent timestamped directory
    timestamp_dirs = [d for d in results_base.iterdir() if d.is_dir() and d.name.replace('_', '').isdigit()]
    
    if not timestamp_dirs:
        print("No timestamped result directories found")
        return
    
    latest_dir = max(timestamp_dirs, key=lambda d: d.name)
    print(f"Analyzing latest enhanced results: {latest_dir}")
    
    analyzer = MetricsAnalyzer(str(latest_dir))
    
    # Generate comprehensive report
    report = analyzer.generate_detailed_report()
    print(report)
    
    # Create enhanced visualizations
    analyzer.create_enhanced_visualizations()


if __name__ == "__main__":
    analyze_latest_results() 