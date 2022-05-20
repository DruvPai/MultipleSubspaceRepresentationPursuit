import torch


def spectrum(A):  # (m, n) -> (min(m, n), )
    return torch.linalg.svdvals(A)  # (min(m, n), )


def projection_onto_col(A):  # (d, n) -> (d, d)
    Q, R = torch.linalg.qr(A)  # (d, d), (d, n)
    U = Q[:, :torch.linalg.matrix_rank(A)]  # (d, r)
    return U @ U.T  # (d, d)


def sin_principal_angle_between_cols(A, B):  # (n, d), (m, d) -> ()
    return torch.linalg.norm(projection_onto_col(A) @ (torch.eye(B.shape[0]) - projection_onto_col(B)), ord=2)  # ()
