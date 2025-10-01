#!/usr/bin/env python3
"""
Memory Snapshot HTML Visualizer
Generates interactive HTML reports for PyTorch memory snapshots
"""

import pickle
import sys
import os
import json
from datetime import datetime


def generate_html_report(snapshot_file, output_file):
    """Generate an interactive HTML report for memory analysis."""
    
    try:
        with open(snapshot_file, 'rb') as f:
            snapshot = pickle.load(f)
        
        if not isinstance(snapshot, dict) or 'segments' not in snapshot:
            print("❌ Invalid snapshot format")
            return
        
        segments = snapshot['segments']
        
        # Prepare data for visualization
        segment_data = []
        total_allocated = 0
        
        for i, segment in enumerate(segments):
            if isinstance(segment, dict):
                allocated_size = segment.get('allocated_size', 0)
                total_size = segment.get('size', 0)
                if allocated_size > 0:
                    segment_data.append({
                        'id': i,
                        'allocated_mb': allocated_size / (1024**2),
                        'total_mb': total_size / (1024**2),
                        'utilization': (allocated_size / total_size * 100) if total_size > 0 else 0
                    })
                    total_allocated += allocated_size
        
        # Get peak memory from device traces
        peak_memory = 0
        device_data = []
        if 'device_traces' in snapshot:
            for dev_i, device_trace in enumerate(snapshot['device_traces']):
                if isinstance(device_trace, list):
                    max_size = 0
                    for trace in device_trace:
                        if isinstance(trace, dict) and 'size' in trace:
                            max_size = max(max_size, trace['size'])
                    device_data.append({
                        'device': dev_i,
                        'peak_mb': max_size / (1024**2)
                    })
                    peak_memory = max(peak_memory, max_size)
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PyTorch Memory Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        .chart-container {{ width: 100%; height: 400px; margin-bottom: 30px; }}
        .segment-list {{ max-height: 300px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .segment-item {{ padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }}
        .segment-item:nth-child(even) {{ background: #f9f9f9; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 PyTorch Memory Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>File:</strong> {os.path.basename(snapshot_file)}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{total_allocated/(1024**3):.2f} GB</div>
                <div class="stat-label">Total Allocated Memory</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len([s for s in segment_data if s['allocated_mb'] > 0])}</div>
                <div class="stat-label">Active Segments</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{peak_memory/(1024**3):.2f} GB</div>
                <div class="stat-label">Peak Memory Usage</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(segments)}</div>
                <div class="stat-label">Total Segments</div>
            </div>
        </div>
        
        <h2>📊 Memory Allocation by Segment</h2>
        <div class="chart-container">
            <canvas id="segmentChart"></canvas>
        </div>
        
        <h2>🖥️ Device Memory Usage</h2>
        <div class="chart-container">
            <canvas id="deviceChart"></canvas>
        </div>
        
        <h2>📋 Top Memory Segments</h2>
        <div class="segment-list">
            {"".join([f'<div class="segment-item"><span>Segment {s["id"]}</span><span>{s["allocated_mb"]:.2f} MB</span></div>' for s in sorted(segment_data, key=lambda x: x["allocated_mb"], reverse=True)[:20]])}
        </div>
        
        <div class="footer">
            <p>Generated by PyTorch Memory Snapshot Analyzer</p>
        </div>
    </div>
    
    <script>
        // Segment allocation chart
        const segmentCtx = document.getElementById('segmentChart').getContext('2d');
        const segmentData = {json.dumps(segment_data[:50])};  // Top 50 segments
        
        new Chart(segmentCtx, {{
            type: 'bar',
            data: {{
                labels: segmentData.map(s => `Segment ${{s.id}}`),
                datasets: [{{
                    label: 'Allocated Memory (MB)',
                    data: segmentData.map(s => s.allocated_mb),
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Memory (MB)'
                        }}
                    }}
                }}
            }}
        }});
        
        // Device memory chart
        const deviceCtx = document.getElementById('deviceChart').getContext('2d');
        const deviceData = {json.dumps(device_data)};
        
        new Chart(deviceCtx, {{
            type: 'doughnut',
            data: {{
                labels: deviceData.map(d => `Device ${{d.device}}`),
                datasets: [{{
                    label: 'Peak Memory (MB)',
                    data: deviceData.map(d => d.peak_mb),
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 205, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)',
                        'rgba(255, 159, 64, 0.8)',
                        'rgba(199, 199, 199, 0.8)',
                        'rgba(83, 102, 255, 0.8)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});
    </script>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report generated: {output_file}")
        print(f"📂 Open in browser: file://{os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python memory_visualizer.py <path_to_pickle_file>")
        sys.exit(1)
    
    snapshot_file = sys.argv[1]
    output_file = os.path.splitext(snapshot_file)[0] + "_report.html"
    
    generate_html_report(snapshot_file, output_file)


if __name__ == "__main__":
    main()