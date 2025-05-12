import torch


def linear_cka(x, y):
    assert (
        x.shape[0] == y.shape[0]
    ), "The feature matrices must contain the same number of samples"
    n_samples = x.shape[0]
    centering_matrix = torch.eye(n_samples) - 1 / n_samples * torch.outer(
        torch.ones(n_samples), torch.ones(n_samples)
    )
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    return float(
        torch.linalg.norm(x.T @ centering_matrix @ y, "fro") ** 2
        / (
            torch.linalg.norm(x.T @ centering_matrix @ x, "fro")
            * torch.linalg.norm(y.T @ centering_matrix @ y, "fro")
        )
    )


if __name__ == "__main__":
    x = torch.rand((10, 5))
    y = torch.rand((10, 4))
    print(linear_cka(x, y))
