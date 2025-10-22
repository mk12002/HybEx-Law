# Create new file: advanced_evaluator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AdvancedEvaluator:
    """
    Comprehensive evaluation with:
    1. Stratified analysis (by income, category, vulnerability)
    2. Confidence calibration curves
    3. Error analysis with examples
    4. Fairness metrics
    5. Production-ready monitoring
    """
    
    def __init__(self, output_dir='results/evaluation_advanced'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def comprehensive_evaluation(self, predictions, ground_truth, metadata):
        """
        Run full evaluation suite.
        
        Args:
            predictions: List of {'eligible': bool, 'confidence': float, 'method': str}
            ground_truth: List of true labels
            metadata: List of dicts with sample metadata (income, category, etc.)
        """
        
        results = {
            'overall_metrics': self._compute_overall_metrics(predictions, ground_truth),
            'stratified_analysis': self._stratified_analysis(predictions, ground_truth, metadata),
            'confidence_analysis': self._analyze_confidence_calibration(predictions, ground_truth),
            'error_analysis': self._detailed_error_analysis(predictions, ground_truth, metadata),
            'fairness_metrics': self._compute_fairness_metrics(predictions, ground_truth, metadata),
            'method_comparison': self._compare_methods(predictions, ground_truth),
        }
        
        # Generate visualizations
        self._generate_visualizations(results, predictions, ground_truth)
        
        # Save report
        self._save_comprehensive_report(results)
        
        return results
    
    def _compute_overall_metrics(self, preds, truth):
        """Compute detailed overall metrics."""
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            matthews_corrcoef, cohen_kappa_score
        )
        
        pred_labels = [p['eligible'] for p in preds]
        
        metrics = {
            'accuracy': accuracy_score(truth, pred_labels),
            'precision': precision_score(truth, pred_labels),
            'recall': recall_score(truth, pred_labels),
            'f1_score': f1_score(truth, pred_labels),
            'mcc': matthews_corrcoef(truth, pred_labels),  # Better for imbalanced data
            'kappa': cohen_kappa_score(truth, pred_labels),  # Agreement score
        }
        
        # Per-class metrics
        tn, fp, fn, tp = confusion_matrix(truth, pred_labels).ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Balanced accuracy (better for imbalanced datasets)
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        return metrics
    
    def _stratified_analysis(self, preds, truth, metadata):
        """Analyze performance across different strata."""
        
        strata = {
            'income_level': [],
            'social_category': [],
            'vulnerability_status': [],
            'has_income': []
        }
        
        # Organize by strata
        for i, (pred, true_label, meta) in enumerate(zip(preds, truth, metadata)):
            # Income level stratification
            income = meta.get('income', 0)
            if income == 0:
                income_level = 'no_income'
            elif income < 15000:
                income_level = 'very_low'
            elif income < 25000:
                income_level = 'low'
            elif income < 40000:
                income_level = 'medium'
            else:
                income_level = 'high'
            
            strata['income_level'].append({
                'stratum': income_level,
                'pred': pred['eligible'],
                'truth': true_label,
                'confidence': pred['confidence']
            })
            
            # Social category stratification
            category = meta.get('social_category', 'general')
            strata['social_category'].append({
                'stratum': category,
                'pred': pred['eligible'],
                'truth': true_label,
                'confidence': pred['confidence']
            })
            
            # Vulnerability status
            is_vulnerable = any([
                meta.get('is_disabled', False),
                meta.get('is_senior_citizen', False),
                meta.get('is_widow', False),
                meta.get('gender') == 'female'
            ])
            strata['vulnerability_status'].append({
                'stratum': 'vulnerable' if is_vulnerable else 'not_vulnerable',
                'pred': pred['eligible'],
                'truth': true_label,
                'confidence': pred['confidence']
            })
            
            # Has income vs no income
            has_income = income > 0
            strata['has_income'].append({
                'stratum': 'has_income' if has_income else 'no_income',
                'pred': pred['eligible'],
                'truth': true_label,
                'confidence': pred['confidence']
            })
        
        # Compute metrics for each stratum
        stratified_results = {}
        
        for dimension, data in strata.items():
            df = pd.DataFrame(data)
            grouped = df.groupby('stratum')
            
            dimension_results = {}
            for stratum_name, group in grouped:
                accuracy = (group['pred'] == group['truth']).mean()
                precision = ((group['pred'] == True) & (group['truth'] == True)).sum() / \
                           max(1, (group['pred'] == True).sum())
                recall = ((group['pred'] == True) & (group['truth'] == True)).sum() / \
                        max(1, (group['truth'] == True).sum())
                
                dimension_results[stratum_name] = {
                    'count': len(group),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'avg_confidence': float(group['confidence'].mean())
                }
            
            stratified_results[dimension] = dimension_results
        
        return stratified_results
    
    def _analyze_confidence_calibration(self, preds, truth):
        """
        Analyze confidence calibration.
        Well-calibrated: predictions with 80% confidence should be correct 80% of time.
        """
        
        # Bin predictions by confidence
        bins = np.linspace(0, 1, 11)  # 10 bins: [0-0.1, 0.1-0.2, ..., 0.9-1.0]
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            bin_lower, bin_upper = bins[i], bins[i + 1]
            
            # Find predictions in this confidence bin
            in_bin = []
            for pred, true_label in zip(preds, truth):
                conf = pred['confidence']
                if bin_lower <= conf < bin_upper or (i == len(bins) - 2 and conf == 1.0):
                    in_bin.append((pred['eligible'], true_label))
            
            if len(in_bin) > 0:
                accuracy = sum(1 for p, t in in_bin if p == t) / len(in_bin)
                avg_conf = (bin_lower + bin_upper) / 2
                
                bin_accs.append(accuracy)
                bin_confs.append(avg_conf)
                bin_counts.append(len(in_bin))
            else:
                bin_accs.append(None)
                bin_confs.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        # Calculate Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = sum(bin_counts)
        
        for acc, conf, count in zip(bin_accs, bin_confs, bin_counts):
            if acc is not None and count > 0:
                ece += (count / total_samples) * abs(acc - conf)
        
        return {
            'bin_accuracies': bin_accs,
            'bin_confidences': bin_confs,
            'bin_counts': bin_counts,
            'expected_calibration_error': float(ece),
            'calibration_quality': 'excellent' if ece < 0.05 else ('good' if ece < 0.10 else 'needs_improvement')
        }
    
    def _detailed_error_analysis(self, preds, truth, metadata, top_k=20):
        """Analyze errors in detail with examples."""
        
        errors = {
            'false_positives': [],  # Predicted eligible, actually not
            'false_negatives': []   # Predicted not eligible, actually eligible
        }
        
        for i, (pred, true_label, meta) in enumerate(zip(preds, truth, metadata)):
            pred_label = pred['eligible']
            
            if pred_label and not true_label:
                # False positive
                errors['false_positives'].append({
                    'index': i,
                    'confidence': pred['confidence'],
                    'method': pred['method'],
                    'income': meta.get('income', 'N/A'),
                    'social_category': meta.get('social_category', 'N/A'),
                    'is_vulnerable': any([
                        meta.get('is_disabled', False),
                        meta.get('is_senior_citizen', False),
                        meta.get('gender') == 'female'
                    ]),
                    'query_snippet': meta.get('query', '')[:100]
                })
            
            elif not pred_label and true_label:
                # False negative
                errors['false_negatives'].append({
                    'index': i,
                    'confidence': pred['confidence'],
                    'method': pred['method'],
                    'income': meta.get('income', 'N/A'),
                    'social_category': meta.get('social_category', 'N/A'),
                    'is_vulnerable': any([
                        meta.get('is_disabled', False),
                        meta.get('is_senior_citizen', False),
                        meta.get('gender') == 'female'
                    ]),
                    'query_snippet': meta.get('query', '')[:100]
                })
        
        # Sort by confidence (high confidence errors are worse)
        errors['false_positives'].sort(key=lambda x: x['confidence'], reverse=True)
        errors['false_negatives'].sort(key=lambda x: x['confidence'], reverse=True)
        
        # Take top K for analysis
        analysis = {
            'false_positives': {
                'count': len(errors['false_positives']),
                'top_errors': errors['false_positives'][:top_k],
                'common_patterns': self._find_error_patterns(errors['false_positives'])
            },
            'false_negatives': {
                'count': len(errors['false_negatives']),
                'top_errors': errors['false_negatives'][:top_k],
                'common_patterns': self._find_error_patterns(errors['false_negatives'])
            }
        }
        
        return analysis
    
    def _find_error_patterns(self, errors):
        """Find common patterns in errors."""
        
        if not errors:
            return {}
        
        patterns = {
            'by_method': {},
            'by_income_range': {},
            'by_category': {},
            'vulnerable_involved': 0
        }
        
        for error in errors:
            # By method
            method = error['method']
            patterns['by_method'][method] = patterns['by_method'].get(method, 0) + 1
            
            # By income range
            income = error.get('income', 'N/A')
            if isinstance(income, (int, float)):
                if income == 0:
                    income_range = 'no_income'
                elif income < 15000:
                    income_range = '<15k'
                elif income < 25000:
                    income_range = '15k-25k'
                elif income < 40000:
                    income_range = '25k-40k'
                else:
                    income_range = '>40k'
            else:
                income_range = 'unknown'
            
            patterns['by_income_range'][income_range] = \
                patterns['by_income_range'].get(income_range, 0) + 1
            
            # By category
            category = error.get('social_category', 'unknown')
            patterns['by_category'][category] = \
                patterns['by_category'].get(category, 0) + 1
            
            # Vulnerable
            if error.get('is_vulnerable', False):
                patterns['vulnerable_involved'] += 1
        
        return patterns
    
    def _compute_fairness_metrics(self, preds, truth, metadata):
        """
        Compute fairness metrics across protected groups.
        Ensure system doesn't discriminate.
        """
        
        fairness = {}
        
        # Group by protected attributes
        groups = {
            'social_category': {},
            'gender': {},
            'vulnerability_status': {}
        }
        
        for pred, true_label, meta in zip(preds, truth, metadata):
            # Social category
            category = meta.get('social_category', 'general')
            if category not in groups['social_category']:
                groups['social_category'][category] = {'preds': [], 'truth': []}
            groups['social_category'][category]['preds'].append(pred['eligible'])
            groups['social_category'][category]['truth'].append(true_label)
            
            # Gender
            gender = meta.get('gender', 'unknown')
            if gender not in groups['gender']:
                groups['gender'][gender] = {'preds': [], 'truth': []}
            groups['gender'][gender]['preds'].append(pred['eligible'])
            groups['gender'][gender]['truth'].append(true_label)
            
            # Vulnerability
            is_vulnerable = any([
                meta.get('is_disabled', False),
                meta.get('is_senior_citizen', False),
                meta.get('is_widow', False)
            ])
            vul_status = 'vulnerable' if is_vulnerable else 'not_vulnerable'
            if vul_status not in groups['vulnerability_status']:
                groups['vulnerability_status'][vul_status] = {'preds': [], 'truth': []}
            groups['vulnerability_status'][vul_status]['preds'].append(pred['eligible'])
            groups['vulnerability_status'][vul_status]['truth'].append(true_label)
        
        # Calculate metrics for each group
        for dimension, group_data in groups.items():
            fairness[dimension] = {}
            
            for group_name, data in group_data.items():
                if len(data['preds']) > 0:
                    # True Positive Rate (Recall)
                    tpr = sum(1 for p, t in zip(data['preds'], data['truth']) 
                             if p and t) / max(1, sum(data['truth']))
                    
                    # False Positive Rate
                    fpr = sum(1 for p, t in zip(data['preds'], data['truth']) 
                             if p and not t) / max(1, len(data['truth']) - sum(data['truth']))
                    
                    fairness[dimension][group_name] = {
                        'count': len(data['preds']),
                        'true_positive_rate': float(tpr),
                        'false_positive_rate': float(fpr),
                        'selection_rate': float(sum(data['preds']) / len(data['preds']))
                    }
        
        # Compute disparate impact (selection rate ratio)
        fairness['disparate_impact'] = self._compute_disparate_impact(fairness)
        
        return fairness
    
    def _compute_disparate_impact(self, fairness):
        """
        Calculate disparate impact ratio.
        Ratio should be > 0.8 to be considered fair (80% rule).
        """
        
        impact = {}
        
        # Social category: compare SC/ST/OBC to general
        if 'social_category' in fairness:
            general_rate = fairness['social_category'].get('general', {}).get('selection_rate', 0)
            
            if general_rate > 0:
                for category, metrics in fairness['social_category'].items():
                    if category != 'general':
                        ratio = metrics['selection_rate'] / general_rate
                        impact[f'{category}_vs_general'] = {
                            'ratio': float(ratio),
                            'fair': ratio >= 0.8,
                            'category_rate': metrics['selection_rate'],
                            'general_rate': general_rate
                        }
        
        return impact
    
    def _compare_methods(self, preds, truth):
        """Compare performance across different methods (BERT, Prolog, Ensemble)."""
        
        method_metrics = {}
        
        # Group by method
        by_method = {}
        for pred, true_label in zip(preds, truth):
            method = pred['method']
            if method not in by_method:
                by_method[method] = {'preds': [], 'truth': [], 'confidences': []}
            
            by_method[method]['preds'].append(pred['eligible'])
            by_method[method]['truth'].append(true_label)
            by_method[method]['confidences'].append(pred['confidence'])
        
        # Calculate metrics for each method
        for method, data in by_method.items():
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            method_metrics[method] = {
                'count': len(data['preds']),
                'accuracy': float(accuracy_score(data['truth'], data['preds'])),
                'precision': float(precision_score(data['truth'], data['preds'])),
                'recall': float(recall_score(data['truth'], data['preds'])),
                'f1_score': float(f1_score(data['truth'], data['preds'])),
                'avg_confidence': float(np.mean(data['confidences']))
            }
        
        return method_metrics
    
    def _generate_visualizations(self, results, preds, truth):
        """Generate comprehensive visualization suite."""
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(preds, truth)
        
        # 2. Calibration Curve
        self._plot_calibration_curve(results['confidence_analysis'])
        
        # 3. Performance by Stratum
        self._plot_stratified_performance(results['stratified_analysis'])
        
        # 4. Method Comparison
        self._plot_method_comparison(results['method_comparison'])
        
        logger.info(f"✅ Visualizations saved to {self.output_dir}")
    
    def _plot_confusion_matrix(self, preds, truth):
        """Plot confusion matrix."""
        pred_labels = [p['eligible'] for p in preds]
        cm = confusion_matrix(truth, pred_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Eligible', 'Eligible'],
                    yticklabels=['Not Eligible', 'Eligible'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
    
    def _plot_calibration_curve(self, calib_data):
        """Plot confidence calibration curve."""
        bin_accs = [a for a in calib_data['bin_accuracies'] if a is not None]
        bin_confs = [c for c, a in zip(calib_data['bin_confidences'], 
                                       calib_data['bin_accuracies']) if a is not None]
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(bin_confs, bin_accs, 'o-', label='Model Calibration')
        plt.xlabel('Predicted Confidence')
        plt.ylabel('Actual Accuracy')
        plt.title(f'Calibration Curve (ECE: {calib_data["expected_calibration_error"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration_curve.png', dpi=300)
        plt.close()
    
    def _plot_stratified_performance(self, stratified):
        """Plot performance across different strata."""
        
        # Income level performance
        if 'income_level' in stratified:
            data = stratified['income_level']
            
            strata_names = list(data.keys())
            accuracies = [data[s]['accuracy'] for s in strata_names]
            counts = [data[s]['count'] for s in strata_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Accuracy by income level
            ax1.bar(strata_names, accuracies)
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Income Level')
            ax1.set_title('Accuracy by Income Level')
            ax1.set_ylim([0, 1])
            for i, (acc, count) in enumerate(zip(accuracies, counts)):
                ax1.text(i, acc + 0.02, f'n={count}', ha='center')
            
            # Sample distribution
            ax2.bar(strata_names, counts)
            ax2.set_ylabel('Sample Count')
            ax2.set_xlabel('Income Level')
            ax2.set_title('Sample Distribution')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'stratified_income.png', dpi=300)
            plt.close()
    
    def _plot_method_comparison(self, method_data):
        """Plot comparison across methods."""
        
        methods = list(method_data.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(methods))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [method_data[m][metric] for m in methods]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_ylabel('Score')
        ax.set_xlabel('Method')
        ax.set_title('Performance Comparison Across Methods')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'method_comparison.png', dpi=300)
        plt.close()
    
    def _compare_methods(self, preds, truth):
        """Compare performance across different methods (BERT, Prolog, Ensemble)."""
        
        method_metrics = {}
        
        # Group by method
        by_method = {}
        for pred, true_label in zip(preds, truth):
            method = pred.get('method', 'unknown')
            if method not in by_method:
                by_method[method] = {'preds': [], 'truth': [], 'confidences': []}
            
            by_method[method]['preds'].append(pred['eligible'])
            by_method[method]['truth'].append(true_label)
            by_method[method]['confidences'].append(pred['confidence'])
        
        # Calculate metrics for each method
        for method, data in by_method.items():
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            if len(data['preds']) > 0:
                method_metrics[method] = {
                    'count': len(data['preds']),
                    'accuracy': float(accuracy_score(data['truth'], data['preds'])),
                    'precision': float(precision_score(data['truth'], data['preds'], zero_division=0)),
                    'recall': float(recall_score(data['truth'], data['preds'], zero_division=0)),
                    'f1_score': float(f1_score(data['truth'], data['preds'], zero_division=0)),
                    'avg_confidence': float(np.mean(data['confidences']))
                }
        
        return method_metrics
    
    def _generate_visualizations(self, results, preds, truth):
        """Generate comprehensive visualization suite."""
        try:
            self._plot_confusion_matrix(preds, truth)
            self._plot_calibration_curve(results['confidence_analysis'])
            if 'stratified_analysis' in results:
                self._plot_stratified_performance(results['stratified_analysis'])
            if 'method_comparison' in results:
                self._plot_method_comparison(results['method_comparison'])
            logger.info(f"✅ Visualizations saved to {self.output_dir}")
        except Exception as e:
            logger.error(f"Visualization error: {e}")
    
    def _plot_confusion_matrix(self, preds, truth):
        """Plot confusion matrix."""
        pred_labels = [p['eligible'] for p in preds]
        cm = confusion_matrix(truth, pred_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Eligible', 'Eligible'],
                    yticklabels=['Not Eligible', 'Eligible'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, calib_data):
        """Plot confidence calibration curve."""
        bin_accs = [a for a in calib_data['bin_accuracies'] if a is not None]
        bin_confs = [c for c, a in zip(calib_data['bin_confidences'], 
                                       calib_data['bin_accuracies']) if a is not None]
        
        if len(bin_accs) > 0:
            plt.figure(figsize=(8, 6))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            plt.plot(bin_confs, bin_accs, 'o-', label='Model Calibration')
            plt.xlabel('Predicted Confidence')
            plt.ylabel('Actual Accuracy')
            plt.title(f'Calibration Curve (ECE: {calib_data["expected_calibration_error"]:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_stratified_performance(self, stratified):
        """Plot performance across different strata."""
        if 'income_level' in stratified:
            data = stratified['income_level']
            strata_names = list(data.keys())
            accuracies = [data[s]['accuracy'] for s in strata_names]
            counts = [data[s]['count'] for s in strata_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.bar(strata_names, accuracies)
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Income Level')
            ax1.set_title('Accuracy by Income Level')
            ax1.set_ylim([0, 1])
            for i, (acc, count) in enumerate(zip(accuracies, counts)):
                ax1.text(i, acc + 0.02, f'n={count}', ha='center')
            
            ax2.bar(strata_names, counts)
            ax2.set_ylabel('Sample Count')
            ax2.set_xlabel('Income Level')
            ax2.set_title('Sample Distribution')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'stratified_income.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_method_comparison(self, method_data):
        """Plot comparison across methods."""
        methods = list(method_data.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(methods))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [method_data[m][metric] for m in methods]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_ylabel('Score')
        ax.set_xlabel('Method')
        ax.set_title('Performance Comparison Across Methods')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_comprehensive_report(self, results):
        """Save detailed report as JSON and markdown."""
        
        # Save JSON
        with open(self.output_dir / 'evaluation_report.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate markdown report
        report_lines = [
            "# Comprehensive Evaluation Report\n",
            f"\nGenerated: {pd.Timestamp.now()}\n",
            "\n## Overall Metrics\n",
            "\n```"
        ]
        
        for metric, value in results['overall_metrics'].items():
            if isinstance(value, float):
                report_lines.append(f"{metric}: {value:.4f}")
            else:
                report_lines.append(f"{metric}: {value}")
        
        report_lines.append("```\n")
        
        # Add other sections
        report_lines.extend([
            "\n## Confidence Calibration\n",
            f"- Expected Calibration Error: {results['confidence_analysis']['expected_calibration_error']:.4f}",
            f"- Quality: {results['confidence_analysis']['calibration_quality']}\n",
            "\n## Error Analysis\n",
            f"- False Positives: {results['error_analysis']['false_positives']['count']}",
            f"- False Negatives: {results['error_analysis']['false_negatives']['count']}\n",
        ])
        
        # Save markdown
        with open(self.output_dir / 'evaluation_report.md', 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"✅ Comprehensive report saved to {self.output_dir}")
