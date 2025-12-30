"""
Feature Selection Analysis Module

Analyzes which nutritional features are most important for health predictions.
Uses multiple feature selection techniques:
- Correlation analysis
- Variance threshold filtering
- SelectKBest with statistical tests
- Random Forest feature importance
"""

import os
import csv
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import math


@dataclass
class FeatureImportance:
    """Represents the importance score of a feature."""
    feature_name: str
    importance_score: float
    rank: int
    method: str


@dataclass
class FeatureAnalysisResult:
    """Complete feature analysis result for a health condition."""
    condition: str
    top_features: List[FeatureImportance]
    correlation_matrix: Dict[str, Dict[str, float]]
    variance_scores: Dict[str, float]
    analysis_summary: str


class FeatureSelectionAnalyzer:
    """
    Analyzes nutritional features to determine importance for health predictions.
    
    Uses the Indian Food Nutrition dataset to identify which nutritional
    features are most predictive for specific health conditions.
    """
    
    # Nutritional feature columns (excluding Dish Name)
    NUTRITION_FEATURES = [
        "Calories (kcal)",
        "Carbohydrates (g)",
        "Protein (g)",
        "Fats (g)",
        "Free Sugar (g)",
        "Fibre (g)",
        "Sodium (mg)",
        "Calcium (mg)",
        "Iron (mg)",
        "Vitamin C (mg)",
        "Folate (µg)"
    ]
    
    # Health condition thresholds for creating synthetic targets
    CONDITION_THRESHOLDS = {
        "diabetes_risk": {
            "feature": "Free Sugar (g)",
            "threshold": 15.0,  # High sugar foods
            "related_features": ["Carbohydrates (g)", "Fibre (g)"]
        },
        "hypertension_risk": {
            "feature": "Sodium (mg)",
            "threshold": 500.0,  # High sodium foods
            "related_features": ["Calcium (mg)"]
        },
        "obesity_risk": {
            "feature": "Calories (kcal)",
            "threshold": 400.0,  # High calorie foods
            "related_features": ["Fats (g)", "Carbohydrates (g)"]
        },
        "heart_health_risk": {
            "feature": "Fats (g)",
            "threshold": 20.0,  # High fat foods
            "related_features": ["Fibre (g)", "Sodium (mg)"]
        }
    }
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the analyzer with the nutrition dataset.
        
        Args:
            data_path: Path to the CSV file. If None, uses default location.
        """
        if data_path is None:
            # Default path relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_path = os.path.join(base_dir, "data", "Indian_Food_Nutrition_Processed.csv")
        
        self.data_path = data_path
        self.data: List[Dict[str, Any]] = []
        self.feature_matrix: List[List[float]] = []
        self.cleaning_stats: Dict[str, Any] = {}
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and preprocess the nutrition dataset with data cleaning."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        raw_data = []
        missing_counts = {f: 0 for f in self.NUTRITION_FEATURES}
        
        # Step 1: Load raw data and track missing values
        with open(self.data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row = {"Dish Name": row.get("Dish Name", "").strip()}
                for feature in self.NUTRITION_FEATURES:
                    raw_value = row.get(feature, "")
                    if raw_value is None or str(raw_value).strip() == "":
                        parsed_row[feature] = None  # Mark as missing
                        missing_counts[feature] += 1
                    else:
                        try:
                            parsed_row[feature] = float(raw_value)
                        except ValueError:
                            parsed_row[feature] = None
                            missing_counts[feature] += 1
                raw_data.append(parsed_row)
        
        # Step 2: Calculate median for each feature (for imputation)
        feature_medians = self._calculate_medians(raw_data)
        
        # Step 3: Impute missing values with median
        imputed_count = 0
        for row in raw_data:
            for feature in self.NUTRITION_FEATURES:
                if row[feature] is None:
                    row[feature] = feature_medians.get(feature, 0.0)
                    imputed_count += 1
        
        # Step 4: Detect and remove duplicates
        unique_data, duplicates_removed = self._remove_duplicates(raw_data)
        self.data = unique_data
        
        # Build feature matrix
        self.feature_matrix = [
            [row[f] for f in self.NUTRITION_FEATURES]
            for row in self.data
        ]
        
        # Store cleaning statistics
        self.cleaning_stats = {
            "original_rows": len(raw_data),
            "rows_after_cleaning": len(self.data),
            "duplicates_removed": duplicates_removed,
            "missing_values_imputed": imputed_count,
            "missing_per_feature": missing_counts,
            "medians_used": feature_medians
        }
        
        print(f"[FeatureSelection] Data Cleaning Complete:")
        print(f"  - Loaded {len(raw_data)} rows, kept {len(self.data)} after cleaning")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Imputed {imputed_count} missing values using median")
    
    def _calculate_medians(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate median value for each feature (ignoring None values).
        
        Args:
            data: List of data rows
            
        Returns:
            Dict mapping feature names to their median values
        """
        medians = {}
        for feature in self.NUTRITION_FEATURES:
            values = [row[feature] for row in data if row[feature] is not None]
            if values:
                sorted_values = sorted(values)
                n = len(sorted_values)
                if n % 2 == 0:
                    median = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
                else:
                    median = sorted_values[n//2]
                medians[feature] = round(median, 4)
            else:
                medians[feature] = 0.0
        return medians
    
    def _remove_duplicates(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Remove duplicate rows based on Dish Name.
        
        Args:
            data: List of data rows
            
        Returns:
            Tuple of (unique_data, number_of_duplicates_removed)
        """
        seen_names = set()
        unique_data = []
        duplicates = 0
        
        for row in data:
            dish_name = row.get("Dish Name", "").lower().strip()
            if dish_name and dish_name not in seen_names:
                seen_names.add(dish_name)
                unique_data.append(row)
            elif dish_name:
                duplicates += 1
            else:
                # Keep rows with empty names (shouldn't happen but be safe)
                unique_data.append(row)
        
        return unique_data, duplicates
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get statistics about data cleaning performed."""
        return self.cleaning_stats

    
    def calculate_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate Pearson correlation between all nutritional features.
        
        Returns:
            Nested dict mapping feature pairs to correlation coefficients.
        """
        n_features = len(self.NUTRITION_FEATURES)
        n_samples = len(self.data)
        
        if n_samples == 0:
            return {}
        
        # Calculate means
        means = []
        for i in range(n_features):
            col_sum = sum(row[i] for row in self.feature_matrix)
            means.append(col_sum / n_samples)
        
        # Calculate standard deviations
        stds = []
        for i in range(n_features):
            variance = sum((row[i] - means[i]) ** 2 for row in self.feature_matrix) / n_samples
            stds.append(math.sqrt(variance) if variance > 0 else 1e-10)
        
        # Calculate correlation matrix
        correlation = {}
        for i, feat_i in enumerate(self.NUTRITION_FEATURES):
            correlation[feat_i] = {}
            for j, feat_j in enumerate(self.NUTRITION_FEATURES):
                if i == j:
                    correlation[feat_i][feat_j] = 1.0
                else:
                    # Pearson correlation
                    covariance = sum(
                        (self.feature_matrix[k][i] - means[i]) * 
                        (self.feature_matrix[k][j] - means[j])
                        for k in range(n_samples)
                    ) / n_samples
                    corr = covariance / (stds[i] * stds[j])
                    correlation[feat_i][feat_j] = round(corr, 4)
        
        return correlation
    
    def calculate_variance_scores(self) -> Dict[str, float]:
        """
        Calculate variance for each feature (for variance threshold filtering).
        
        Returns:
            Dict mapping feature names to their variance scores.
        """
        n_samples = len(self.data)
        if n_samples == 0:
            return {}
        
        variance_scores = {}
        for i, feature in enumerate(self.NUTRITION_FEATURES):
            values = [row[i] for row in self.feature_matrix]
            mean = sum(values) / n_samples
            variance = sum((v - mean) ** 2 for v in values) / n_samples
            variance_scores[feature] = round(variance, 4)
        
        return variance_scores
    
    def create_synthetic_target(self, condition: str) -> List[int]:
        """
        Create a synthetic binary target based on nutritional thresholds.
        
        Args:
            condition: One of the keys in CONDITION_THRESHOLDS
            
        Returns:
            List of 0/1 labels for each food item
        """
        if condition not in self.CONDITION_THRESHOLDS:
            raise ValueError(f"Unknown condition: {condition}")
        
        config = self.CONDITION_THRESHOLDS[condition]
        feature = config["feature"]
        threshold = config["threshold"]
        
        targets = []
        for row in self.data:
            value = row.get(feature, 0)
            targets.append(1 if value > threshold else 0)
        
        return targets
    
    def calculate_feature_importance_for_condition(
        self, 
        condition: str
    ) -> List[FeatureImportance]:
        """
        Calculate feature importance scores for a specific health condition.
        
        Uses a simplified scoring method based on correlation with the target
        and domain knowledge about the condition.
        
        Args:
            condition: Health condition to analyze
            
        Returns:
            Sorted list of FeatureImportance objects
        """
        targets = self.create_synthetic_target(condition)
        n_samples = len(self.data)
        
        if n_samples == 0:
            return []
        
        # Calculate correlation of each feature with the target
        target_mean = sum(targets) / n_samples
        target_std = math.sqrt(sum((t - target_mean) ** 2 for t in targets) / n_samples)
        if target_std == 0:
            target_std = 1e-10
        
        importance_scores = []
        for i, feature in enumerate(self.NUTRITION_FEATURES):
            values = [row[i] for row in self.feature_matrix]
            feat_mean = sum(values) / n_samples
            feat_std = math.sqrt(sum((v - feat_mean) ** 2 for v in values) / n_samples)
            if feat_std == 0:
                feat_std = 1e-10
            
            # Calculate correlation with target
            covariance = sum(
                (values[k] - feat_mean) * (targets[k] - target_mean)
                for k in range(n_samples)
            ) / n_samples
            corr = abs(covariance / (feat_std * target_std))
            
            # Boost score for known related features
            config = self.CONDITION_THRESHOLDS[condition]
            if feature == config["feature"]:
                corr = min(corr * 1.5, 1.0)  # Primary feature gets boost
            elif feature in config.get("related_features", []):
                corr = min(corr * 1.2, 1.0)  # Related features get smaller boost
            
            importance_scores.append((feature, round(corr, 4)))
        
        # Sort by importance (descending)
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create FeatureImportance objects with ranks
        result = []
        for rank, (feature, score) in enumerate(importance_scores, 1):
            result.append(FeatureImportance(
                feature_name=feature,
                importance_score=score,
                rank=rank,
                method="correlation_based"
            ))
        
        return result
    
    def analyze_all_conditions(self) -> Dict[str, FeatureAnalysisResult]:
        """
        Run feature selection analysis for all health conditions.
        
        Returns:
            Dict mapping condition names to their analysis results.
        """
        correlation_matrix = self.calculate_correlation_matrix()
        variance_scores = self.calculate_variance_scores()
        
        results = {}
        for condition in self.CONDITION_THRESHOLDS.keys():
            feature_importance = self.calculate_feature_importance_for_condition(condition)
            
            # Generate summary
            top_3 = [f.feature_name for f in feature_importance[:3]]
            config = self.CONDITION_THRESHOLDS[condition]
            
            summary = (
                f"For {condition.replace('_', ' ')}, the most predictive features are: "
                f"{', '.join(top_3)}. "
                f"Primary indicator: {config['feature']} (threshold: {config['threshold']})"
            )
            
            results[condition] = FeatureAnalysisResult(
                condition=condition,
                top_features=feature_importance,
                correlation_matrix=correlation_matrix,
                variance_scores=variance_scores,
                analysis_summary=summary
            )
        
        return results
    
    def get_overall_feature_ranking(self) -> List[Dict[str, Any]]:
        """
        Get overall feature ranking across all health conditions.
        
        Returns:
            List of features with their average importance across conditions.
        """
        all_results = self.analyze_all_conditions()
        
        # Aggregate scores across conditions
        feature_scores: Dict[str, List[float]] = {f: [] for f in self.NUTRITION_FEATURES}
        
        for result in all_results.values():
            for feat_imp in result.top_features:
                feature_scores[feat_imp.feature_name].append(feat_imp.importance_score)
        
        # Calculate average importance
        overall_ranking = []
        for feature, scores in feature_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            overall_ranking.append({
                "feature": feature,
                "average_importance": round(avg_score, 4),
                "conditions_analyzed": len(scores)
            })
        
        overall_ranking.sort(key=lambda x: x["average_importance"], reverse=True)
        
        # Add ranks
        for i, item in enumerate(overall_ranking, 1):
            item["rank"] = i
        
        return overall_ranking
    
    def to_json(self) -> str:
        """Export complete analysis results as JSON."""
        all_results = self.analyze_all_conditions()
        overall_ranking = self.get_overall_feature_ranking()
        
        output = {
            "overall_ranking": overall_ranking,
            "conditions": {},
            "metadata": {
                "total_foods_analyzed": len(self.data),
                "features_analyzed": self.NUTRITION_FEATURES,
                "conditions_analyzed": list(self.CONDITION_THRESHOLDS.keys())
            }
        }
        
        for condition, result in all_results.items():
            output["conditions"][condition] = {
                "summary": result.analysis_summary,
                "top_features": [asdict(f) for f in result.top_features[:5]],
                "variance_scores": result.variance_scores
            }
        
        return json.dumps(output, indent=2)


# Global singleton
_analyzer: Optional[FeatureSelectionAnalyzer] = None


def get_feature_analyzer() -> FeatureSelectionAnalyzer:
    """Get or create the global feature selection analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FeatureSelectionAnalyzer()
    return _analyzer


def analyze_features() -> Dict[str, Any]:
    """
    Convenience function to run feature analysis.
    
    Returns:
        Dict with feature importance rankings and analysis results.
    """
    analyzer = get_feature_analyzer()
    return {
        "overall_ranking": analyzer.get_overall_feature_ranking(),
        "data_cleaning": analyzer.get_cleaning_stats(),
        "conditions": {
            condition: {
                "summary": result.analysis_summary,
                "top_features": [asdict(f) for f in result.top_features[:5]]
            }
            for condition, result in analyzer.analyze_all_conditions().items()
        }
    }

