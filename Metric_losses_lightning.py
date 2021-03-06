import torch
import torch.nn as nn

class batch_hard_triplet_loss(torch.nn.Module):

    def __init__(self, margin_negative, squared=False):
        super(batch_hard_triplet_loss, self).__init__()
        self.margin_negative = margin_negative
        self.squared = squared

    def _get_anchor_negative_triplet_mask(self):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        return ~(self.labels.unsqueeze(0) == self.labels.unsqueeze(1))

    def _get_anchor_positive_triplet_mask(self):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = torch.eye(self.labels.size(0)).bool().to(self.embeddings.device)
        indices_not_equal = ~indices_equal

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = self.labels.unsqueeze(0) == self.labels.unsqueeze(1)

        return labels_equal & indices_not_equal

    def _get_triplet_mask(self):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal = torch.eye(self.labels.size(0)).bool()
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

        label_equal = self.labels.unsqueeze(0) == self.labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = ~i_equal_k & i_equal_j

        return valid_labels & distinct_indices

    def _pairwise_distances(self):
        """Compute the 2D matrix of distances between all the embeddings.
          Args:
              embeddings: tensor of shape (batch_size, embed_dim)
              squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                       If false, output is the pairwise euclidean distance matrix.
          Returns:
              pairwise_distances: tensor of shape (batch_size, batch_size)
          """
        dot_product = torch.matmul(self.embeddings, self.embeddings.t())
        # Get squared L@ norm for each emnbedding. We can just take the diagonal of the dot_product
        # This also provides more numerical stability (the diagonal of the result will be eactly 0)
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a-b||^2 = ||a||^2 -2<a,b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so put everyhting >= 0.0
        distances[distances < 0] = 0

        if not self.squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex. on the diagonal)
            # we need to add a small epsilon where distances = 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16
            distances = (1.0 - mask) * torch.sqrt(distances)

        return distances

    def forward(self, labels, embeddings):
        self.embeddings = embeddings

        self.labels = labels
        """Build the triplet loss over a batch of embeddings.
            For each anchor, we get the hardest positive and hardest negative to form a triplet.
            Args:
                labels: labels of the batch, of size (batch_size,)
                embeddings: tensor of shape (batch_size, embed_dim)
                margin: margin for triplet loss
                squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                         If false, output is the pairwise euclidean distance matrix.
            Returns:
                triplet_loss: scalar tensor containing the triplet loss
            """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances()

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask()
        #print(mask_anchor_positive.size())

        # print(mask_anchor_positive)
        # print(pairwise_dist)
        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask().float()
        #print(mask_anchor_negative.size())

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        #print(hardest_positive_dist)
        #print(hardest_negative_dist)
        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + self.margin_negative

        #t2 = hardest_positive_dist - (self.margin_negative/2)

        #correct_negative = tl[tl < 0].numel()
        #correct_positive = t2[t2 < 0].numel()
        tl[tl < 0] = 0
        #t2[t2 < 0] = 0
        #print(tl)
        triplet_loss = tl.mean() #+ t2.mean()

        return triplet_loss, hardest_positive_dist.mean(), hardest_negative_dist.mean()


