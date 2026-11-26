import unittest
from cosine_diversity import cosine_summary


class CosineTests(unittest.TestCase):
    def test_duplicates_have_no_dissimilar_peers(self):
        rows = cosine_summary([[1,0]]*32)
        self.assertTrue(all(r['peer_count']==31 and r['fraction_similarity_below_threshold']==0 for r in rows))

    def test_self_is_excluded(self):
        rows = cosine_summary([[1,0],[0,1]])
        self.assertTrue(all(r['nearest_cosine_similarity']==0 and r['fraction_similarity_below_threshold']==1 for r in rows))

    def test_duplicates_are_visible_even_with_distinct_outlier(self):
        rows = cosine_summary([[1,0],[1,0],[0,1]])
        self.assertEqual(rows[0]['nearest_cosine_similarity'],1)
        self.assertEqual(rows[0]['fraction_similarity_below_threshold'],0.5)
        self.assertEqual(rows[2]['fraction_similarity_below_threshold'],1)

    def test_scaling_does_not_change_similarity(self):
        self.assertEqual(cosine_summary([[3,0],[0,4]]),cosine_summary([[1,0],[0,1]]))

    def test_invalid_embeddings_fail(self):
        for embeddings in ([[0,0],[1,0]], [[1],[1,0]], [[float('nan')],[1]], [[1]], [[],[]]):
            with self.subTest(embeddings=embeddings), self.assertRaises(ValueError):
                cosine_summary(embeddings)


if __name__ == '__main__':
    unittest.main()
