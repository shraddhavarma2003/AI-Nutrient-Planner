"""
Tests for Feature Selection Module
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestFeatureSelectionAnalyzer:
    """Tests for the FeatureSelectionAnalyzer class."""
    
    def test_data_loading(self):
        """Test that nutrition data is loaded correctly."""
        from analytics.feature_selection import FeatureSelectionAnalyzer
        
        analyzer = FeatureSelectionAnalyzer()
        
        # Should have loaded data
        assert len(analyzer.data) > 0, "Should load data from CSV"
        assert len(analyzer.feature_matrix) > 0, "Should build feature matrix"
        
        # Check first item has expected features
        first_item = analyzer.data[0]
        assert "Dish Name" in first_item
        assert "Calories (kcal)" in first_item
        assert "Sodium (mg)" in first_item
    
    def test_correlation_matrix(self):
        """Test correlation matrix calculation."""
        from analytics.feature_selection import FeatureSelectionAnalyzer
        
        analyzer = FeatureSelectionAnalyzer()
        correlation = analyzer.calculate_correlation_matrix()
        
        # Should have all features
        assert len(correlation) == len(analyzer.NUTRITION_FEATURES)
        
        # Diagonal should be 1.0 (self-correlation)
        for feature in analyzer.NUTRITION_FEATURES:
            assert feature in correlation
            assert correlation[feature][feature] == 1.0
        
        # Correlations should be between -1 and 1
        for feat_i, row in correlation.items():
            for feat_j, value in row.items():
                assert -1.0 <= value <= 1.0, f"Correlation out of range: {feat_i} vs {feat_j}"
    
    def test_variance_scores(self):
        """Test variance calculation for features."""
        from analytics.feature_selection import FeatureSelectionAnalyzer
        
        analyzer = FeatureSelectionAnalyzer()
        variance = analyzer.calculate_variance_scores()
        
        # Should have variance for all features
        assert len(variance) == len(analyzer.NUTRITION_FEATURES)
        
        # All variances should be non-negative
        for feature, score in variance.items():
            assert score >= 0, f"Variance should be non-negative: {feature}"
    
    def test_synthetic_target_creation(self):
        """Test synthetic target creation for health conditions."""
        from analytics.feature_selection import FeatureSelectionAnalyzer
        
        analyzer = FeatureSelectionAnalyzer()
        
        # Test diabetes risk target
        targets = analyzer.create_synthetic_target("diabetes_risk")
        assert len(targets) == len(analyzer.data)
        assert all(t in [0, 1] for t in targets), "Targets should be binary"
        
        # Should have some positive and negative examples
        assert sum(targets) > 0, "Should have some high-risk foods"
        assert sum(targets) < len(targets), "Should have some low-risk foods"
    
    def test_feature_importance_ranking(self):
        """Test feature importance calculation."""
        from analytics.feature_selection import FeatureSelectionAnalyzer
        
        analyzer = FeatureSelectionAnalyzer()
        
        importance = analyzer.calculate_feature_importance_for_condition("diabetes_risk")
        
        # Should rank all features
        assert len(importance) == len(analyzer.NUTRITION_FEATURES)
        
        # Should be sorted by rank
        ranks = [f.rank for f in importance]
        assert ranks == list(range(1, len(importance) + 1))
        
        # Sugar should be highly ranked for diabetes
        sugar_rank = next(f.rank for f in importance if "Sugar" in f.feature_name)
        assert sugar_rank <= 3, "Sugar should be in top 3 for diabetes risk"
    
    def test_analyze_all_conditions(self):
        """Test analysis across all conditions."""
        from analytics.feature_selection import FeatureSelectionAnalyzer
        
        analyzer = FeatureSelectionAnalyzer()
        results = analyzer.analyze_all_conditions()
        
        # Should have results for all conditions
        expected_conditions = ["diabetes_risk", "hypertension_risk", "obesity_risk", "heart_health_risk"]
        for condition in expected_conditions:
            assert condition in results, f"Missing condition: {condition}"
            assert results[condition].top_features is not None
            assert results[condition].analysis_summary is not None
    
    def test_overall_ranking(self):
        """Test overall feature ranking across conditions."""
        from analytics.feature_selection import FeatureSelectionAnalyzer
        
        analyzer = FeatureSelectionAnalyzer()
        ranking = analyzer.get_overall_feature_ranking()
        
        # Should rank all features
        assert len(ranking) == len(analyzer.NUTRITION_FEATURES)
        
        # Each item should have required fields
        for item in ranking:
            assert "feature" in item
            assert "average_importance" in item
            assert "rank" in item
    
    def test_json_export(self):
        """Test JSON export functionality."""
        import json
        from analytics.feature_selection import FeatureSelectionAnalyzer
        
        analyzer = FeatureSelectionAnalyzer()
        json_output = analyzer.to_json()
        
        # Should be valid JSON
        data = json.loads(json_output)
        
        assert "overall_ranking" in data
        assert "conditions" in data
        assert "metadata" in data


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_feature_analyzer(self):
        """Test singleton analyzer creation."""
        from analytics.feature_selection import get_feature_analyzer
        
        analyzer1 = get_feature_analyzer()
        analyzer2 = get_feature_analyzer()
        
        # Should return same instance (singleton)
        assert analyzer1 is analyzer2
    
    def test_analyze_features(self):
        """Test the analyze_features convenience function."""
        from analytics.feature_selection import analyze_features
        
        result = analyze_features()
        
        assert "overall_ranking" in result
        assert "conditions" in result
        
        # Check structure of conditions
        for condition, data in result["conditions"].items():
            assert "summary" in data
            assert "top_features" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
