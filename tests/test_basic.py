"""Basic tests for CNSBM package."""

import pytest
import numpy as np
import jax.numpy as jnp


def test_import():
    """Test that package imports correctly."""
    try:
        from cnsbm import CNSBM, CNSBMTrainer
        assert CNSBM is not None
        assert CNSBMTrainer is not None
    except ImportError as e:
        pytest.fail(f"Failed to import CNSBM: {e}")


def test_version():
    """Test that version is accessible."""
    try:
        import cnsbm
        assert hasattr(cnsbm, "__version__")
        assert isinstance(cnsbm.__version__, str)
    except ImportError as e:
        pytest.fail(f"Failed to import cnsbm: {e}")


@pytest.fixture
def simple_data():
    """Generate simple test data."""
    np.random.seed(42)
    return jnp.asarray(np.random.randint(0, 3, size=(20, 15)))


def test_cnsbm_initialization(simple_data):
    """Test CNSBM model initialization."""
    from cnsbm import CNSBM
    
    model = CNSBM(simple_data, K=3, L=2)
    assert model.N == 20
    assert model.M == 15
    assert model.K == 3
    assert model.L == 2
    assert model.num_cat == 3


def test_cnsbm_basic_functionality(simple_data):
    """Test basic CNSBM functionality."""
    from cnsbm import CNSBM
    
    model = CNSBM(simple_data, K=2, L=2)
    
    # Test that model can run a few iterations without crashing
    model.batch_vi(3, batch_print=1)
    
    # Check that model is marked as fitted
    assert model.fitted
    
    # Check that training history is recorded
    assert len(model.training_history) > 0
    
    # Test summary method
    model.summary()


if __name__ == "__main__":
    pytest.main([__file__])
