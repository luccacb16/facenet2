import torch

def semi_hard_triplet_mining(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    device: str = 'cuda',
    hardest: bool = True
) -> torch.Tensor:
    
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    num_embeddings = embeddings.size(0)

    # Create a mask for positive pairs
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_pos_pairs = labels_equal.logical_xor(torch.eye(num_embeddings, device=device, dtype=torch.bool))

    # Find hardest positive for each anchor
    hardest_positive_dist, _ = (distance_matrix + (~mask_pos_pairs).float() * 1e6).min(dim=1)

    # Create a mask for semi-hard negatives
    mask_semi_hard = ((distance_matrix > hardest_positive_dist.unsqueeze(1)) & 
                      (distance_matrix < hardest_positive_dist.unsqueeze(1) + margin))
    mask_neg_pairs = ~labels_equal

    mask_semi_hard_or_hard = mask_semi_hard | (~mask_semi_hard & mask_neg_pairs)

    if hardest:
        # Find hardest negative among semi-hard or all negatives
        _, hardest_negative_indices = (distance_matrix + (~mask_semi_hard_or_hard).float() * 1e6).min(dim=1)
    else:
        # Randomly select from semi-hard negatives or all negatives if no semi-hard exist
        valid_negative_indices = mask_semi_hard_or_hard.nonzero(as_tuple=True)
        perm = torch.randperm(valid_negative_indices[0].size(0), device=device)
        hardest_negative_indices = valid_negative_indices[1][perm[:num_embeddings]]

    # Randomly select positive
    valid_positive_indices = mask_pos_pairs.nonzero(as_tuple=True)
    perm = torch.randperm(valid_positive_indices[0].size(0), device=device)
    random_positive_indices = valid_positive_indices[1][perm[:num_embeddings]]

    triplets = torch.stack([torch.arange(num_embeddings, device=device), 
                            random_positive_indices, 
                            hardest_negative_indices], dim=1)

    return triplets

def hard_negative_triplet_mining(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    device: str = 'cuda'
) -> torch.Tensor:
    
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    num_embeddings = embeddings.size(0)

    # Create a mask for positive pairs
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_pos_pairs = labels_equal.logical_xor(torch.eye(num_embeddings, device=device, dtype=torch.bool))
    mask_neg_pairs = ~labels_equal

    # Randomly select positive
    valid_positive_indices = mask_pos_pairs.nonzero(as_tuple=True)
    perm = torch.randperm(valid_positive_indices[0].size(0), device=device)
    random_positive_indices = valid_positive_indices[1][perm[:num_embeddings]]

    # Find hardest negative
    _, hardest_negative_indices = (distance_matrix + (~mask_neg_pairs).float() * 1e6).min(dim=1)

    triplets = torch.stack([torch.arange(num_embeddings, device=device), 
                            random_positive_indices, 
                            hardest_negative_indices], dim=1)

    return triplets