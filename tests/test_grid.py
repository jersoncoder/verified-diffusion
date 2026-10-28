import numpy as np
import pytest

from verified_diffusion.grid import Grid1D, Grid2D


def test_uniform_grid_geometry() -> None:
    g = Grid1D.uniform(0.0, 2.0, 4)
    np.testing.assert_allclose(g.faces, [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(g.centres, [0.25, 0.75, 1.25, 1.75])
    np.testing.assert_allclose(g.widths, 0.5)
    assert g.n == 4
    assert g.length == pytest.approx(2.0)
    assert g.max_width_ratio == pytest.approx(1.0)


@pytest.mark.parametrize("cluster", ["left", "right", "both"])
def test_stretched_grid_spans_interval_and_clusters(cluster: str) -> None:
    g = Grid1D.stretched(1.0, 3.0, 40, beta=2.0, cluster=cluster)  # type: ignore[arg-type]
    assert g.faces[0] == 1.0
    assert g.faces[-1] == 3.0
    assert np.all(g.widths > 0.0)
    assert g.widths.sum() == pytest.approx(2.0, rel=1e-14)
    assert g.max_width_ratio > 2.0
    small = int(np.argmin(g.widths))
    if cluster == "left":
        assert small == 0
    elif cluster == "right":
        assert small == g.n - 1
    else:
        assert small in (0, g.n - 1)


def test_mapping_grids_nest_under_refinement() -> None:
    coarse = Grid1D.stretched(0.0, 1.0, 10, beta=1.5)
    fine = Grid1D.stretched(0.0, 1.0, 20, beta=1.5)
    np.testing.assert_allclose(fine.faces[::2], coarse.faces, rtol=0, atol=1e-15)


def test_stretched_grid_widths_vary_smoothly() -> None:
    # Adjacent-width ratios tend to 1 as the grid is refined: the property that
    # keeps the cell-centred scheme second order on stretched grids.
    ratios = []
    for n in (20, 40, 80):
        w = Grid1D.stretched(0.0, 1.0, n, beta=2.5).widths
        ratios.append(float(np.max(w[1:] / w[:-1])) - 1.0)
    assert ratios[1] < 0.55 * ratios[0]
    assert ratios[2] < 0.55 * ratios[1]


@pytest.mark.parametrize(
    "faces",
    [[0.0], [[0.0, 1.0]], [0.0, 0.0, 1.0], [1.0, 0.0], [0.0, np.nan]],
)
def test_invalid_faces_are_rejected(faces: object) -> None:
    with pytest.raises(ValueError, match="faces"):
        Grid1D(faces)  # type: ignore[arg-type]


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        Grid1D.uniform(0.0, 1.0, 0)
    with pytest.raises(ValueError, match="beta"):
        Grid1D.stretched(0.0, 1.0, 10, beta=0.0)
    with pytest.raises(ValueError, match="cluster"):
        Grid1D.stretched(0.0, 1.0, 10, beta=1.0, cluster="middle")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mapping"):
        Grid1D.from_mapping(0.0, 1.0, 4, lambda xi: 2.0 * xi)


def test_grid_arrays_are_read_only() -> None:
    g = Grid1D.uniform(0.0, 1.0, 3)
    with pytest.raises(ValueError, match="read-only"):
        g.faces[0] = 5.0


def test_grid2d_tensor_product() -> None:
    g = Grid2D(Grid1D.uniform(0.0, 2.0, 4), Grid1D.stretched(0.0, 1.0, 3, beta=1.0))
    assert g.shape == (4, 3)
    assert g.size == 12
    X, Y = g.centres()
    assert X.shape == Y.shape == (4, 3)
    np.testing.assert_allclose(X[:, 0], g.x.centres)
    np.testing.assert_allclose(Y[0, :], g.y.centres)
    assert g.volumes().sum() == pytest.approx(2.0)
