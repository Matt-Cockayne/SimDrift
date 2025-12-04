"""
Alert system for drift and performance degradation.

Features:
- Configurable thresholds for alerts
- Multi-level severity (info, warning, critical)
- Remediation recommendations
- Alert history tracking
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertSystem:
    """
    Alert system for model monitoring.
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.2,
        performance_threshold: float = 0.05,
        calibration_threshold: float = 0.1
    ):
        """
        Initialize alert system.
        
        Args:
            drift_threshold: Threshold for drift detection alerts
            performance_threshold: Threshold for performance degradation (e.g., 5% drop)
            calibration_threshold: Threshold for calibration degradation
        """
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.calibration_threshold = calibration_threshold
        
        self.alerts = []
        self.alert_count = {'info': 0, 'warning': 0, 'critical': 0}
    
    def check_drift_alert(
        self,
        drift_results: Dict,
        timestamp: Optional[str] = None
    ) -> List[Dict]:
        """
        Check drift detection results and generate alerts.
        
        Args:
            drift_results: Results from DriftDetector
            timestamp: Optional timestamp
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        if not drift_results['drift_detected']:
            return alerts
        
        # Check each method
        for method, data in drift_results['methods'].items():
            if data.get('drift_detected', False):
                severity = self._determine_drift_severity(method, data)
                
                alert = {
                    'type': 'drift',
                    'severity': severity.value,
                    'method': method,
                    'timestamp': timestamp or datetime.now().isoformat(),
                    'message': self._generate_drift_message(method, data),
                    'data': data,
                    'recommendations': self._get_drift_recommendations(method, severity)
                }
                
                alerts.append(alert)
                self.alerts.append(alert)
                self.alert_count[severity.value] += 1
        
        return alerts
    
    def check_performance_alert(
        self,
        current_metrics: Dict,
        reference_metrics: Optional[Dict] = None,
        timestamp: Optional[str] = None
    ) -> List[Dict]:
        """
        Check performance metrics and generate alerts.
        
        Args:
            current_metrics: Current performance metrics
            reference_metrics: Reference performance metrics for comparison
            timestamp: Optional timestamp
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        if reference_metrics is None:
            # No comparison, just check absolute thresholds
            return self._check_absolute_performance(current_metrics, timestamp)
        
        # Compare performance
        classification = current_metrics.get('classification', {})
        ref_classification = reference_metrics.get('classification', {})
        
        for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
            if metric in classification and metric in ref_classification:
                current = classification[metric]
                reference = ref_classification[metric]
                
                degradation = reference - current
                relative_degradation = degradation / reference if reference > 0 else 0
                
                if degradation > self.performance_threshold:
                    severity = self._determine_performance_severity(relative_degradation)
                    
                    alert = {
                        'type': 'performance',
                        'severity': severity.value,
                        'metric': metric,
                        'timestamp': timestamp or datetime.now().isoformat(),
                        'message': f"{metric} degraded by {degradation:.4f} ({relative_degradation*100:.1f}%)",
                        'current_value': current,
                        'reference_value': reference,
                        'degradation': degradation,
                        'recommendations': self._get_performance_recommendations(metric, severity)
                    }
                    
                    alerts.append(alert)
                    self.alerts.append(alert)
                    self.alert_count[severity.value] += 1
        
        return alerts
    
    def check_calibration_alert(
        self,
        current_calibration: Dict,
        reference_calibration: Optional[Dict] = None,
        timestamp: Optional[str] = None
    ) -> List[Dict]:
        """
        Check calibration metrics and generate alerts.
        
        Args:
            current_calibration: Current calibration metrics
            reference_calibration: Reference calibration for comparison
            timestamp: Optional timestamp
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        current_ece = current_calibration.get('ece', 0)
        
        # Check absolute ECE threshold
        if current_ece > self.calibration_threshold:
            severity = AlertSeverity.WARNING if current_ece < 0.2 else AlertSeverity.CRITICAL
            
            alert = {
                'type': 'calibration',
                'severity': severity.value,
                'metric': 'ece',
                'timestamp': timestamp or datetime.now().isoformat(),
                'message': f"Expected Calibration Error (ECE) is {current_ece:.4f}",
                'current_value': current_ece,
                'threshold': self.calibration_threshold,
                'recommendations': self._get_calibration_recommendations(severity)
            }
            
            alerts.append(alert)
            self.alerts.append(alert)
            self.alert_count[severity.value] += 1
        
        # Check degradation if reference provided
        if reference_calibration is not None:
            ref_ece = reference_calibration.get('ece', 0)
            degradation = current_ece - ref_ece
            
            if degradation > 0.05:
                severity = AlertSeverity.WARNING if degradation < 0.1 else AlertSeverity.CRITICAL
                
                alert = {
                    'type': 'calibration_drift',
                    'severity': severity.value,
                    'metric': 'ece',
                    'timestamp': timestamp or datetime.now().isoformat(),
                    'message': f"Calibration degraded by {degradation:.4f}",
                    'current_value': current_ece,
                    'reference_value': ref_ece,
                    'degradation': degradation,
                    'recommendations': self._get_calibration_recommendations(severity)
                }
                
                alerts.append(alert)
                self.alerts.append(alert)
                self.alert_count[severity.value] += 1
        
        return alerts
    
    def check_fairness_alert(
        self,
        fairness_metrics: Dict,
        timestamp: Optional[str] = None
    ) -> List[Dict]:
        """
        Check fairness metrics and generate alerts.
        
        Args:
            fairness_metrics: Fairness metrics from PerformanceTracker
            timestamp: Optional timestamp
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Thresholds for fairness
        parity_threshold = 0.1  # 10% difference
        odds_threshold = 0.1
        
        for attr_name, metrics in fairness_metrics.items():
            # Check demographic parity
            parity_diff = metrics.get('demographic_parity_diff', 0)
            if parity_diff > parity_threshold:
                severity = AlertSeverity.WARNING if parity_diff < 0.2 else AlertSeverity.CRITICAL
                
                alert = {
                    'type': 'fairness',
                    'severity': severity.value,
                    'attribute': attr_name,
                    'metric': 'demographic_parity',
                    'timestamp': timestamp or datetime.now().isoformat(),
                    'message': f"Demographic parity violation for {attr_name}: {parity_diff:.4f}",
                    'value': parity_diff,
                    'threshold': parity_threshold,
                    'recommendations': self._get_fairness_recommendations(attr_name, 'demographic_parity')
                }
                
                alerts.append(alert)
                self.alerts.append(alert)
                self.alert_count[severity.value] += 1
            
            # Check equalized odds (if available)
            odds_diff = metrics.get('equalized_odds_diff', 0)
            if odds_diff > odds_threshold:
                severity = AlertSeverity.WARNING if odds_diff < 0.2 else AlertSeverity.CRITICAL
                
                alert = {
                    'type': 'fairness',
                    'severity': severity.value,
                    'attribute': attr_name,
                    'metric': 'equalized_odds',
                    'timestamp': timestamp or datetime.now().isoformat(),
                    'message': f"Equalized odds violation for {attr_name}: {odds_diff:.4f}",
                    'value': odds_diff,
                    'threshold': odds_threshold,
                    'recommendations': self._get_fairness_recommendations(attr_name, 'equalized_odds')
                }
                
                alerts.append(alert)
                self.alerts.append(alert)
                self.alert_count[severity.value] += 1
        
        return alerts
    
    def _determine_drift_severity(self, method: str, data: Dict) -> AlertSeverity:
        """Determine severity of drift alert."""
        if method == 'psi':
            max_psi = max(data['scores'])
            if max_psi > 0.5:
                return AlertSeverity.CRITICAL
            elif max_psi > 0.3:
                return AlertSeverity.WARNING
            return AlertSeverity.INFO
        
        elif method in ['ks', 'chi2']:
            min_pval = min(data['p_values'])
            if min_pval < 0.001:
                return AlertSeverity.CRITICAL
            elif min_pval < 0.01:
                return AlertSeverity.WARNING
            return AlertSeverity.INFO
        
        elif method == 'mmd':
            mmd = data['score']
            if mmd > 0.3:
                return AlertSeverity.CRITICAL
            elif mmd > 0.15:
                return AlertSeverity.WARNING
            return AlertSeverity.INFO
        
        return AlertSeverity.INFO
    
    def _determine_performance_severity(self, relative_degradation: float) -> AlertSeverity:
        """Determine severity of performance degradation."""
        if relative_degradation > 0.15:  # >15% degradation
            return AlertSeverity.CRITICAL
        elif relative_degradation > 0.08:  # >8% degradation
            return AlertSeverity.WARNING
        return AlertSeverity.INFO
    
    def _generate_drift_message(self, method: str, data: Dict) -> str:
        """Generate human-readable drift message."""
        if method == 'psi':
            max_psi = max(data['scores'])
            return f"Data drift detected (PSI: {max_psi:.4f}). Distribution shift exceeds threshold."
        
        elif method == 'ks':
            min_pval = min(data['p_values'])
            return f"Data drift detected (KS test p-value: {min_pval:.4f}). Distributions significantly different."
        
        elif method == 'chi2':
            min_pval = min(data['p_values'])
            return f"Data drift detected (Chi-square p-value: {min_pval:.4f}). Feature distributions changed."
        
        elif method == 'mmd':
            mmd = data['score']
            return f"Data drift detected (MMD: {mmd:.4f}). Multivariate distribution shift detected."
        
        return "Data drift detected."
    
    def _get_drift_recommendations(self, method: str, severity: AlertSeverity) -> List[str]:
        """Get recommendations for drift remediation."""
        recommendations = [
            "📊 Investigate feature distributions for significant changes",
            "🔍 Check data collection pipeline for issues",
            "📈 Review recent changes in data sources or preprocessing"
        ]
        
        if severity == AlertSeverity.CRITICAL:
            recommendations.extend([
                "🚨 URGENT: Consider model retraining immediately",
                "⚠️ Evaluate current model predictions for reliability",
                "🔄 Implement temporary fallback mechanisms if available"
            ])
        elif severity == AlertSeverity.WARNING:
            recommendations.extend([
                "⏰ Plan model retraining within next maintenance cycle",
                "📉 Monitor performance metrics closely",
                "🧪 Test model on recent data samples"
            ])
        
        return recommendations
    
    def _get_performance_recommendations(self, metric: str, severity: AlertSeverity) -> List[str]:
        """Get recommendations for performance degradation."""
        recommendations = [
            f"📉 {metric} has degraded significantly",
            "🔍 Investigate potential causes (data drift, concept drift, etc.)",
            "📊 Analyze error patterns and failure modes"
        ]
        
        if severity == AlertSeverity.CRITICAL:
            recommendations.extend([
                "🚨 URGENT: Model retraining required",
                "🛑 Consider pausing automated decisions",
                "👥 Route predictions to human review temporarily"
            ])
        else:
            recommendations.extend([
                "📅 Schedule model update",
                "🧹 Review and clean training data",
                "⚙️ Consider hyperparameter tuning"
            ])
        
        return recommendations
    
    def _get_calibration_recommendations(self, severity: AlertSeverity) -> List[str]:
        """Get recommendations for calibration issues."""
        recommendations = [
            "🎯 Model predictions are poorly calibrated",
            "📊 Review confidence scores vs. actual accuracy",
            "🔧 Consider calibration techniques (temperature scaling, isotonic regression)"
        ]
        
        if severity == AlertSeverity.CRITICAL:
            recommendations.append("⚠️ Do not trust prediction confidence scores for decision-making")
        
        return recommendations
    
    def _get_fairness_recommendations(self, attribute: str, metric: str) -> List[str]:
        """Get recommendations for fairness violations."""
        return [
            f"⚖️ Fairness violation detected for {attribute}",
            "👥 Review model predictions across demographic groups",
            "📊 Analyze training data for representation bias",
            "🔄 Consider fairness-aware training techniques",
            "📝 Document fairness concerns for stakeholders",
            "🎯 Implement bias mitigation strategies"
        ]
    
    def _check_absolute_performance(self, metrics: Dict, timestamp: Optional[str]) -> List[Dict]:
        """Check performance against absolute thresholds."""
        alerts = []
        classification = metrics.get('classification', {})
        
        # Minimum acceptable performance
        min_accuracy = 0.7
        
        if 'accuracy' in classification:
            accuracy = classification['accuracy']
            if accuracy < min_accuracy:
                severity = AlertSeverity.CRITICAL if accuracy < 0.5 else AlertSeverity.WARNING
                
                alert = {
                    'type': 'performance',
                    'severity': severity.value,
                    'metric': 'accuracy',
                    'timestamp': timestamp or datetime.now().isoformat(),
                    'message': f"Model accuracy ({accuracy:.4f}) below minimum threshold ({min_accuracy})",
                    'current_value': accuracy,
                    'threshold': min_accuracy,
                    'recommendations': self._get_performance_recommendations('accuracy', severity)
                }
                
                alerts.append(alert)
                self.alerts.append(alert)
                self.alert_count[severity.value] += 1
        
        return alerts
    
    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get alerts with optional filtering.
        
        Args:
            severity: Filter by severity
            alert_type: Filter by type ('drift', 'performance', 'calibration', 'fairness')
            limit: Maximum number of alerts to return
            
        Returns:
            List of filtered alerts
        """
        filtered = self.alerts
        
        if severity:
            filtered = [a for a in filtered if a['severity'] == severity.value]
        
        if alert_type:
            filtered = [a for a in filtered if a['type'] == alert_type]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts."""
        return {
            'total_alerts': len(self.alerts),
            'by_severity': dict(self.alert_count),
            'by_type': self._count_by_type(),
            'recent_critical': len([a for a in self.alerts[-10:] if a['severity'] == 'critical'])
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count alerts by type."""
        types = {}
        for alert in self.alerts:
            alert_type = alert['type']
            types[alert_type] = types.get(alert_type, 0) + 1
        return types
    
    def generate_alert_report(self, last_n: int = 10) -> str:
        """
        Generate human-readable alert report.
        
        Args:
            last_n: Number of recent alerts to include
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 70)
        report.append("ALERT REPORT")
        report.append("=" * 70)
        
        summary = self.get_alert_summary()
        report.append(f"\nTotal Alerts: {summary['total_alerts']}")
        report.append(f"  Critical: {summary['by_severity']['critical']}")
        report.append(f"  Warning: {summary['by_severity']['warning']}")
        report.append(f"  Info: {summary['by_severity']['info']}")
        
        report.append(f"\nBy Type:")
        for alert_type, count in summary['by_type'].items():
            report.append(f"  {alert_type}: {count}")
        
        recent_alerts = self.alerts[-last_n:]
        if recent_alerts:
            report.append(f"\nRecent Alerts (last {len(recent_alerts)}):")
            for i, alert in enumerate(recent_alerts, 1):
                severity_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}[alert['severity']]
                report.append(f"\n{i}. {severity_emoji} [{alert['severity'].upper()}] {alert['type']}")
                report.append(f"   {alert['message']}")
                report.append(f"   Time: {alert['timestamp']}")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)
    
    def reset(self):
        """Clear all alerts and reset counters."""
        self.alerts = []
        self.alert_count = {'info': 0, 'warning': 0, 'critical': 0}


if __name__ == '__main__':
    # Example usage
    print("Testing AlertSystem...")
    
    alert_system = AlertSystem()
    
    # Simulate drift alert
    drift_results = {
        'drift_detected': True,
        'methods': {
            'psi': {
                'scores': [0.35, 0.12, 0.08],
                'drift_detected': True,
                'threshold': 0.2
            }
        },
        'feature_drift': {}
    }
    
    alerts = alert_system.check_drift_alert(drift_results, timestamp='2024-01-01')
    print(f"\n✓ Generated {len(alerts)} drift alert(s)")
    
    # Simulate performance alert
    current_metrics = {'classification': {'accuracy': 0.75, 'f1_macro': 0.73}}
    reference_metrics = {'classification': {'accuracy': 0.85, 'f1_macro': 0.83}}
    
    alerts = alert_system.check_performance_alert(current_metrics, reference_metrics, timestamp='2024-01-02')
    print(f"✓ Generated {len(alerts)} performance alert(s)")
    
    # Generate report
    print("\n" + alert_system.generate_alert_report())
    
    print("\n✓ All tests passed!")
