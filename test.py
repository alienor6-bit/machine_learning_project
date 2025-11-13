#!/usr/bin/env python3
"""
Test the robust forex data preparation
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def test_forex_robust():
    """Test the robust forex data preparation"""
    print("🧪 Testing robust forex data preparation...")

    try:
        from forex_data import prepare_forex_dataset
        print("✅ Successfully imported forex_data_robust")

        # Test data preparation
        print("\n📊 Testing data preparation...")
        dataset = prepare_forex_dataset()

        print(f"\n🎉 SUCCESS!")
        print(f"   Dataset shape: {dataset.shape}")
        print(f"   Date range: {dataset.index[0]} to {dataset.index[-1]}")

        # Show columns
        feature_cols = [col for col in dataset.columns if not col.startswith('Target_')]
        target_cols = [col for col in dataset.columns if col.startswith('Target_')]

        print(f"\n📋 Column breakdown:")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Targets: {len(target_cols)}")

        print(f"\n📊 First few feature columns:")
        for i, col in enumerate(feature_cols[:10], 1):
            print(f"   {i:2d}. {col}")

        print(f"\n🎯 Target columns:")
        for i, col in enumerate(target_cols, 1):
            print(f"   {i:2d}. {col}")

        print(f"\n📈 Sample data (last 3 rows):")
        sample_cols = ['Close', 'RSI_14', 'MACD', 'Target_Return_1d', 'Target_Return_5d']
        available_cols = [col for col in sample_cols if col in dataset.columns]
        print(dataset[available_cols].tail(3))

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_forex_robust()

    if success:
        print("\n✅ ROBUST FOREX DATA TEST PASSED!")
        print("🚀 You can now run: python main_forex.py")
    else:
        print("\n❌ Test failed. Check the error messages above.")
