"""
Reporting and Visualization

This module provides comprehensive reporting and visualization capabilities
for benchmark results, including charts, tables, and formatted reports.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report output formats"""
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"


class ChartType(Enum):
    """Chart types for visualization"""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    TABLE = "table"
    HEATMAP = "heatmap"


@dataclass
class ReportSection:
    """
    A section in a report.

    Attributes:
        title: Section title
        content: Section content
        chart_type: Optional chart type
        chart_data: Optional chart data
        metadata: Additional metadata
    """
    title: str
    content: str
    chart_type: Optional[ChartType] = None
    chart_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ReportGenerator:
    """
    Advanced report generation with visualization support.

    This class creates comprehensive reports with:
    - Multiple output formats (text, HTML, markdown, JSON, CSV)
    - Data visualization (charts, tables, heatmaps)
    - Comparative analysis
    - Statistical summaries
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize report generator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.sections: List[ReportSection] = []

    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report"""
        self.sections.append(section)

    def clear_sections(self) -> None:
        """Clear all sections"""
        self.sections = []

    def generate_comprehensive_report(
        self,
        results: Dict[str, Any],
        output_format: ReportFormat = ReportFormat.HTML
    ) -> str:
        """
        Generate comprehensive report from benchmark results.

        Args:
            results: Results dictionary
            output_format: Output format

        Returns:
            Formatted report string
        """
        self.clear_sections()

        # Add summary section
        self._add_summary_section(results)

        # Add detailed metrics section
        self._add_metrics_section(results)

        # Add domain analysis section
        self._add_domain_analysis_section(results)

        # Add complexity analysis section
        self._add_complexity_analysis_section(results)

        # Add baseline comparison section
        if "baselines" in results and results["baselines"]:
            self._add_baseline_comparison_section(results)

        # Add failed problems section
        self._add_failed_problems_section(results)

        # Generate report based on format
        if output_format == ReportFormat.TEXT:
            return self._format_as_text()
        elif output_format == ReportFormat.JSON:
            return self._format_as_json()
        elif output_format == ReportFormat.HTML:
            return self._format_as_html()
        elif output_format == ReportFormat.MARKDOWN:
            return self._format_as_markdown()
        elif output_format == ReportFormat.CSV:
            return self._format_as_csv(results)
        else:
            raise ValueError(f"Unknown output format: {output_format}")

    def _add_summary_section(self, results: Dict[str, Any]) -> None:
        """Add executive summary section"""
        if "main" not in results:
            return

        metrics = results["main"]["metrics"]

        content = f"""
Total Problems: {metrics.total_problems}
Successful: {metrics.successful} ({metrics.success_rate:.1%})
Failed: {metrics.failed}
Average Time: {metrics.avg_time_ms:.2f} ms
Median Time: {metrics.median_time_ms:.2f} ms
"""

        self.add_section(ReportSection(
            title="Executive Summary",
            content=content.strip(),
            metadata={"type": "summary"}
        ))

    def _add_metrics_section(self, results: Dict[str, Any]) -> None:
        """Add detailed metrics section"""
        if "main" not in results:
            return

        metrics = results["main"]["metrics"]

        content = f"""
Performance Metrics:
  Total Time: {metrics.total_time_ms:.2f} ms
  Average Time: {metrics.avg_time_ms:.2f} ms
  Median Time: {metrics.median_time_ms:.2f} ms
  Min Time: {metrics.min_time_ms:.2f} ms
  Max Time: {metrics.max_time_ms:.2f} ms

Success Metrics:
  Total: {metrics.total_problems}
  Successful: {metrics.successful}
  Failed: {metrics.failed}
  Success Rate: {metrics.success_rate:.1%}
"""

        # Confidence metrics if available
        if metrics.confidence_metrics:
            content += "\nConfidence Metrics:\n"
            for key, value in metrics.confidence_metrics.items():
                content += f"  {key}: {value:.3f}\n"

        self.add_section(ReportSection(
            title="Detailed Metrics",
            content=content.strip(),
            chart_type=ChartType.TABLE,
            chart_data={
                "headers": ["Metric", "Value"],
                "rows": [
                    ["Success Rate", f"{metrics.success_rate:.1%}"],
                    ["Avg Time", f"{metrics.avg_time_ms:.2f} ms"],
                    ["Median Time", f"{metrics.median_time_ms:.2f} ms"],
                ]
            }
        ))

    def _add_domain_analysis_section(self, results: Dict[str, Any]) -> None:
        """Add domain-specific analysis section"""
        if "main" not in results:
            return

        metrics = results["main"]["metrics"]

        if not metrics.domain_metrics:
            return

        content = "Domain Performance:\n\n"
        chart_data = {
            "labels": [],
            "success_rates": [],
            "avg_times": [],
        }

        for domain, dm in metrics.domain_metrics.items():
            content += f"{domain}:\n"
            content += f"  Total: {dm['total']}\n"
            content += f"  Success Rate: {dm['success_rate']:.1%}\n"
            content += f"  Avg Time: {dm['avg_time_ms']:.2f} ms\n"
            if dm.get("avg_overall_score") is not None:
                content += f"  Avg Score: {dm['avg_overall_score']:.3f}\n"
            content += "\n"

            chart_data["labels"].append(domain)
            chart_data["success_rates"].append(dm["success_rate"])
            chart_data["avg_times"].append(dm["avg_time_ms"])

        self.add_section(ReportSection(
            title="Domain Analysis",
            content=content.strip(),
            chart_type=ChartType.BAR,
            chart_data=chart_data
        ))

    def _add_complexity_analysis_section(self, results: Dict[str, Any]) -> None:
        """Add complexity-specific analysis section"""
        if "main" not in results:
            return

        metrics = results["main"]["metrics"]

        if not metrics.complexity_metrics:
            return

        content = "Complexity Performance:\n\n"
        chart_data = {
            "labels": [],
            "success_rates": [],
            "avg_times": [],
        }

        # Sort by complexity order
        complexity_order = ["trivial", "simple", "moderate", "complex", "advanced"]
        sorted_complexities = sorted(
            metrics.complexity_metrics.items(),
            key=lambda x: complexity_order.index(x[0]) if x[0] in complexity_order else 999
        )

        for complexity, cm in sorted_complexities:
            content += f"{complexity}:\n"
            content += f"  Total: {cm['total']}\n"
            content += f"  Success Rate: {cm['success_rate']:.1%}\n"
            content += f"  Avg Time: {cm['avg_time_ms']:.2f} ms\n"
            if cm.get("avg_overall_score") is not None:
                content += f"  Avg Score: {cm['avg_overall_score']:.3f}\n"
            content += "\n"

            chart_data["labels"].append(complexity)
            chart_data["success_rates"].append(cm["success_rate"])
            chart_data["avg_times"].append(cm["avg_time_ms"])

        self.add_section(ReportSection(
            title="Complexity Analysis",
            content=content.strip(),
            chart_type=ChartType.LINE,
            chart_data=chart_data
        ))

    def _add_baseline_comparison_section(self, results: Dict[str, Any]) -> None:
        """Add baseline comparison section"""
        if "main" not in results or "baselines" not in results:
            return

        main_metrics = results["main"]["metrics"]
        baselines = results["baselines"]

        content = "Baseline Comparisons:\n\n"
        chart_data = {
            "labels": ["Main"],
            "success_rates": [main_metrics.success_rate],
            "avg_times": [main_metrics.avg_time_ms],
        }

        for baseline_name, baseline_metrics in baselines.items():
            success_diff = main_metrics.success_rate - baseline_metrics.success_rate
            time_diff = baseline_metrics.avg_time_ms - main_metrics.avg_time_ms

            content += f"{baseline_name}:\n"
            content += f"  Success Rate: {baseline_metrics.success_rate:.1%} "
            content += f"({success_diff:+.1%} vs main)\n"
            content += f"  Avg Time: {baseline_metrics.avg_time_ms:.2f} ms "
            content += f"({time_diff:+.2f} ms vs main)\n"
            content += "\n"

            chart_data["labels"].append(baseline_name)
            chart_data["success_rates"].append(baseline_metrics.success_rate)
            chart_data["avg_times"].append(baseline_metrics.avg_time_ms)

        self.add_section(ReportSection(
            title="Baseline Comparisons",
            content=content.strip(),
            chart_type=ChartType.BAR,
            chart_data=chart_data
        ))

    def _add_failed_problems_section(self, results: Dict[str, Any]) -> None:
        """Add failed problems section"""
        if "main" not in results or "results" not in results["main"]:
            return

        failed = [r for r in results["main"]["results"] if not r.success]

        if not failed:
            content = "No failed problems!"
        else:
            content = f"Failed Problems ({len(failed)}):\n\n"
            for result in failed[:10]:  # Limit to first 10
                content += f"- {result.problem_id} ({result.domain}/{result.complexity})\n"
                if result.error:
                    content += f"  Error: {result.error}\n"

            if len(failed) > 10:
                content += f"\n... and {len(failed) - 10} more"

        self.add_section(ReportSection(
            title="Failed Problems",
            content=content.strip(),
            metadata={"failed_count": len(failed)}
        ))

    def _format_as_text(self) -> str:
        """Format report as plain text"""
        lines = []
        lines.append("=" * 80)
        lines.append("TMR BENCHMARK VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        for section in self.sections:
            lines.append(section.title.upper())
            lines.append("-" * 80)
            lines.append(section.content)
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def _format_as_json(self) -> str:
        """Format report as JSON"""
        data = {
            "title": "TMR Benchmark Validation Report",
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "chart_type": section.chart_type.value if section.chart_type else None,
                    "chart_data": section.chart_data,
                    "metadata": section.metadata,
                }
                for section in self.sections
            ]
        }
        return json.dumps(data, indent=2)

    def _format_as_html(self) -> str:
        """Format report as HTML with charts"""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<title>TMR Benchmark Report</title>")
        html.append("<meta charset='utf-8'>")
        html.append(self._get_html_styles())
        html.append("</head>")
        html.append("<body>")
        html.append("<div class='container'>")
        html.append("<h1>TMR Benchmark Validation Report</h1>")

        for section in self.sections:
            html.append(f"<div class='section'>")
            html.append(f"<h2>{section.title}</h2>")
            html.append(f"<pre class='content'>{section.content}</pre>")

            # Add chart if available
            if section.chart_type and section.chart_data:
                html.append(self._generate_html_chart(section.chart_type, section.chart_data))

            html.append("</div>")

        html.append("</div>")
        html.append("</body>")
        html.append("</html>")

        return "\n".join(html)

    def _get_html_styles(self) -> str:
        """Get HTML styles"""
        return """
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
    background-color: white;
    padding: 30px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-radius: 8px;
}
h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}
h2 {
    color: #34495e;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 8px;
    margin-top: 30px;
}
.section {
    margin-bottom: 30px;
}
.content {
    background-color: #f8f9fa;
    padding: 15px;
    border-left: 4px solid #3498db;
    overflow-x: auto;
    font-family: 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.6;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 12px;
    text-align: left;
}
th {
    background-color: #3498db;
    color: white;
    font-weight: bold;
}
tr:nth-child(even) {
    background-color: #f2f2f2;
}
tr:hover {
    background-color: #e8f4f8;
}
.chart {
    margin: 20px 0;
    padding: 15px;
    background-color: #fff;
    border: 1px solid #ddd;
    border-radius: 4px;
}
.bar {
    display: inline-block;
    background-color: #3498db;
    color: white;
    padding: 5px 10px;
    margin: 5px 0;
    border-radius: 3px;
}
</style>
"""

    def _generate_html_chart(self, chart_type: ChartType, chart_data: Dict[str, Any]) -> str:
        """Generate HTML chart (simplified ASCII visualization)"""
        if chart_type == ChartType.TABLE:
            return self._generate_html_table(chart_data)
        elif chart_type == ChartType.BAR:
            return self._generate_html_bar_chart(chart_data)
        else:
            return ""

    def _generate_html_table(self, chart_data: Dict[str, Any]) -> str:
        """Generate HTML table"""
        html = ["<table>"]

        # Headers
        if "headers" in chart_data:
            html.append("<tr>")
            for header in chart_data["headers"]:
                html.append(f"<th>{header}</th>")
            html.append("</tr>")

        # Rows
        if "rows" in chart_data:
            for row in chart_data["rows"]:
                html.append("<tr>")
                for cell in row:
                    html.append(f"<td>{cell}</td>")
                html.append("</tr>")

        html.append("</table>")
        return "\n".join(html)

    def _generate_html_bar_chart(self, chart_data: Dict[str, Any]) -> str:
        """Generate simple HTML bar chart"""
        html = ["<div class='chart'>"]

        if "labels" in chart_data and "success_rates" in chart_data:
            max_rate = max(chart_data["success_rates"]) if chart_data["success_rates"] else 1.0

            for label, rate in zip(chart_data["labels"], chart_data["success_rates"]):
                width = int((rate / max_rate) * 400) if max_rate > 0 else 0
                html.append(f"<div>")
                html.append(f"<strong>{label}:</strong> ")
                html.append(f"<div class='bar' style='width: {width}px;'>{rate:.1%}</div>")
                html.append(f"</div>")

        html.append("</div>")
        return "\n".join(html)

    def _format_as_markdown(self) -> str:
        """Format report as Markdown"""
        lines = []
        lines.append("# TMR Benchmark Validation Report")
        lines.append("")

        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append("```")
            lines.append(section.content)
            lines.append("```")
            lines.append("")

            # Add chart data as table if available
            if section.chart_type == ChartType.TABLE and section.chart_data:
                lines.append(self._generate_markdown_table(section.chart_data))
                lines.append("")

        return "\n".join(lines)

    def _generate_markdown_table(self, chart_data: Dict[str, Any]) -> str:
        """Generate Markdown table"""
        lines = []

        if "headers" in chart_data:
            lines.append("| " + " | ".join(chart_data["headers"]) + " |")
            lines.append("| " + " | ".join(["---"] * len(chart_data["headers"])) + " |")

        if "rows" in chart_data:
            for row in chart_data["rows"]:
                lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)

    def _format_as_csv(self, results: Dict[str, Any]) -> str:
        """Format detailed results as CSV"""
        lines = []

        # Header
        lines.append("Problem ID,Domain,Complexity,Success,Time (ms),Score,Error")

        # Data rows
        if "main" in results and "results" in results["main"]:
            for result in results["main"]["results"]:
                score = result.score.overall if result.score and hasattr(result.score, "overall") else ""
                error = result.error.replace(",", ";") if result.error else ""

                lines.append(
                    f"{result.problem_id},"
                    f"{result.domain},"
                    f"{result.complexity},"
                    f"{result.success},"
                    f"{result.execution_time_ms:.2f},"
                    f"{score},"
                    f"{error}"
                )

        return "\n".join(lines)

    def generate_comparison_report(
        self,
        results_a: Dict[str, Any],
        results_b: Dict[str, Any],
        label_a: str = "Baseline",
        label_b: str = "Current",
        output_format: ReportFormat = ReportFormat.HTML
    ) -> str:
        """
        Generate comparison report between two result sets.

        Args:
            results_a: First results
            results_b: Second results
            label_a: Label for first results
            label_b: Label for second results
            output_format: Output format

        Returns:
            Formatted comparison report
        """
        self.clear_sections()

        # Add comparison summary
        self._add_comparison_summary(results_a, results_b, label_a, label_b)

        # Generate report
        if output_format == ReportFormat.TEXT:
            return self._format_as_text()
        elif output_format == ReportFormat.JSON:
            return self._format_as_json()
        elif output_format == ReportFormat.HTML:
            return self._format_as_html()
        elif output_format == ReportFormat.MARKDOWN:
            return self._format_as_markdown()
        else:
            raise ValueError(f"Unknown output format: {output_format}")

    def _add_comparison_summary(
        self,
        results_a: Dict[str, Any],
        results_b: Dict[str, Any],
        label_a: str,
        label_b: str
    ) -> None:
        """Add comparison summary section"""
        metrics_a = results_a.get("main", {}).get("metrics")
        metrics_b = results_b.get("main", {}).get("metrics")

        if not metrics_a or not metrics_b:
            return

        success_diff = metrics_b.success_rate - metrics_a.success_rate
        time_diff = metrics_a.avg_time_ms - metrics_b.avg_time_ms

        content = f"""
{label_a} vs {label_b}:

Success Rate:
  {label_a}: {metrics_a.success_rate:.1%}
  {label_b}: {metrics_b.success_rate:.1%}
  Difference: {success_diff:+.1%}

Average Time:
  {label_a}: {metrics_a.avg_time_ms:.2f} ms
  {label_b}: {metrics_b.avg_time_ms:.2f} ms
  Difference: {time_diff:+.2f} ms
"""

        self.add_section(ReportSection(
            title="Comparison Summary",
            content=content.strip()
        ))
