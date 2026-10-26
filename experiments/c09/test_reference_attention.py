"""Deterministic mathematical tests; no data/models/dependencies/downloads."""
import math
import random
import unittest
from reference_attention import *


class AttentionTests(unittest.TestCase):
    def close(self, a, b, tol=2e-12):
        self.assertEqual(len(a), len(b))
        self.assertLessEqual(max(abs(x-y) for x,y in zip(a,b)), tol)

    def test_binary64(self):
        self.assertEqual(binary64_environment()["mant_dig"], 53)

    def test_unequal_and_empty_regions(self):
        s = [0.] * 10
        v = [[1.]] + [[-1.]] * 9
        h = hierarchical_attention(s, v, [[], [0], list(range(1,10)), []])
        self.close(h["region_probabilities"], [0., .1, .9, 0.])
        self.close(h["output"], [-.8])
        self.close(dense_attention(s,v)["output"], h["output"])

    def test_conditionals_recover_dense_probability(self):
        s = [2., -1., 5., 0.]
        r = [[0,3], [], [1,2]]
        h = hierarchical_attention(s, [[i] for i in range(4)], r)
        p = dense_attention(s, [[i] for i in range(4)])["probabilities"]
        for region, wr, cond in zip(r, h["region_probabilities"],
                                    h["conditional_probabilities"]):
            for i, a in zip(region, cond):
                self.assertAlmostEqual(wr*a, p[i], places=14)

    def test_wrong_block_average(self):
        h = hierarchical_attention([0.]*10, [[1.]]+[[-1.]]*9,
                                   [[0], list(range(1,10))])
        wrong = sum(x[0] for x in h["region_outputs"])/2
        self.assertAlmostEqual(wrong, 0.)
        self.assertAlmostEqual(abs(wrong-h["output"][0]), .8)

    def test_extreme_logits(self):
        s = [10000., -10000., 9999., -math.inf]
        v = [[1.,0.], [0.,1.], [-1.,0.], [2.,2.]]
        self.close(hierarchical_attention(s,v,[[0,1],[],[2,3]])["output"],
                   dense_attention(s,v)["output"])
        with self.assertRaises(OverflowError):
            math.exp(10000.)

    def test_huge_common_offset(self):
        # Absolute logsumexp([1e300]*9) rounds to 1e300, losing log(9).
        s = [1e300]*10
        v = [[1.]]+[[-1.]]*9
        self.close(hierarchical_attention(s,v,[[0],list(range(1,10))])["output"],
                   dense_attention(s,v)["output"])
        self.close(softmax([logsumexp([s[0]]),logsumexp(s[1:])]), [.5,.5])

    def test_masked_regions_and_empty_lse(self):
        self.assertEqual(logsumexp([]), -math.inf)
        self.assertEqual(logsumexp([-math.inf]), -math.inf)
        h = hierarchical_attention([-math.inf,0.],[[99.],[2.]],[[0],[],[1]])
        self.close(h["region_probabilities"],[0.,0.,1.])
        self.close(h["output"],[2.])

    def test_all_masked_rejected(self):
        with self.assertRaises(ValueError):
            hierarchical_attention([-math.inf],[[1.]],[[0]])
        with self.assertRaises(ValueError):
            softmax([])
        with self.assertRaises(ValueError):
            selected_attention([-math.inf,0.],[[1.],[2.]],[[0],[1]],[0])

    def test_nonfinite_rejected(self):
        for x in [math.nan, math.inf]:
            with self.assertRaises(ValueError):
                softmax([0.,x])
        with self.assertRaises(ValueError):
            dense_attention([0.],[[math.inf]])

    def test_partition_overlap_duplicate_missing(self):
        for r in [[[0],[0,1]], [[0,0],[1]], [[0]], [[-1],[0,1]], [[0],[2]]]:
            with self.assertRaises(ValueError):
                hierarchical_attention([0.,0.],[[1.],[2.]],r)
        with self.assertRaises(ValueError):
            selected_attention([0.,0.],[[1.],[2.]],[[0],[1]],[0,0])
        with self.assertRaises(ValueError):
            partition([[0.]],1)

    def test_all_selected_and_one_region(self):
        s = [0.,1.,2.]
        v = [[1.],[3.],[-1.]]
        r = [[0],[],[1,2]]
        self.close(selected_attention(s,v,r,[0,2])["output"],
                   dense_attention(s,v)["output"])
        self.close(selected_attention(s,v,r,[0])["output"],[1.])
        with self.assertRaises(ValueError):
            selected_attention(s,v,r,[1])
        with self.assertRaises(ValueError):
            select_topk([0.,-math.inf],2)

    def test_mean_key_identity(self):
        q=[2.,-1.]
        keys=[[1.,4.],[3.,2.],[-2.,5.]]
        logits=attention_logits(q,keys)
        centroid=[sum(row[j] for row in keys)/3 for j in range(2)]
        self.assertAlmostEqual(attention_logits(q,[centroid])[0],sum(logits)/3)

    def test_flat_log_size_exact(self):
        s=[2.]+[-1.]*9
        r=[[0],list(range(1,10)),[]]
        exact=softmax(approximate_region_logits(s,r,"logsumexp"))
        self.close(softmax(approximate_region_logits(s,r,"mean_log_size")),exact)
        self.assertGreater(l2([x-y for x,y in zip(
            softmax(approximate_region_logits(s,r,"mean_key")),exact)]), .1)

    def test_max_ranking_fails_cardinality(self):
        s=[3.]+[0.]*64
        r=[[0],list(range(1,65))]
        self.assertEqual(select_topk(approximate_region_logits(s,r,"max"),1),[0])
        self.assertEqual(select_topk(approximate_region_logits(s,r,"logsumexp"),1),[1])

    def test_mean_log_size_misses_peak(self):
        s=[1.,8.]+[-8.]*63
        r=[[0],list(range(1,65))]
        self.assertEqual(select_topk(approximate_region_logits(s,r,"mean_log_size"),1),[0])
        self.assertEqual(select_topk(approximate_region_logits(s,r,"logsumexp"),1),[1])

    def test_duplicate_representation(self):
        self.close(dense_attention([0.,0.],[[1.],[-1.]])["output"],[0.])
        self.close(dense_attention([0.,0.,0.],[[1.],[1.],[-1.]])["output"],[1/3])
        # Known identical copies only: each copy gets -log(multiplicity).
        self.close(dense_attention([-math.log(2),-math.log(2),0.],
                                  [[1.],[1.],[-1.]])["output"],[0.])

    def test_bound_tight(self):
        d=truncation_diagnostic([math.log(.9),math.log(.1)],[[1.],[-1.]],
                                [[0],[1]],[0])
        self.assertAlmostEqual(d["l2_output_error"],.2)
        self.assertAlmostEqual(d["error_bound_2B_epsilon"],.2)

    def test_bound_needs_value_bound(self):
        d=truncation_diagnostic([math.log(1-1e-6),math.log(1e-6)],
                                [[0.],[1e6]],[[0],[1]],[0])
        self.assertAlmostEqual(d["l2_output_error"],1.)
        self.assertGreater(d["l2_output_error"],2e-6)

    def test_bound_needs_exact_conditional(self):
        s=[0.]*10
        v=[[1.]]+[[-1.]]*9
        r=[[0],list(range(1,10))]
        diag=truncation_diagnostic(s,v,r,[0,1])
        self.assertEqual(diag["omitted_mass"],0.)
        # Equal regional mixing changes distribution even with no omission.
        wrong=0.
        self.assertGreater(abs(wrong-dense_attention(s,v)["output"][0]),.79)

    def test_zero_error_despite_omission(self):
        d=truncation_diagnostic([0.,0.],[[2.],[2.]],[[0],[1]],[0])
        self.assertEqual(d["l2_output_error"],0.)
        self.assertAlmostEqual(d["omitted_mass"],.5)

    def test_random_equivalence_and_bounds_up_to_1024(self):
        rng=random.Random(9027)
        for n in [1,2,7,31,127,511,1024]:
            for repeat in range(5):
                s=[rng.uniform(-40,40) for _ in range(n)]
                v=[]
                for i in range(n):
                    row=[rng.gauss(0,1) for _ in range(5)]
                    norm=l2(row)
                    v.append([x/max(1.,norm) for x in row])
                regions=[[] for _ in range(11)]
                for i in range(n):
                    regions[rng.randrange(11)].append(i)
                h=hierarchical_attention(s,v,regions)
                self.close(h["output"],dense_attention(s,v)["output"])
                scores=approximate_region_logits(s,regions,"logsumexp")
                eligible=sum(x != -math.inf for x in scores)
                for k in sorted(set([1,eligible,max(1,eligible//2)])):
                    selected=select_topk(scores,k)
                    d=truncation_diagnostic(s,v,regions,selected)
                    self.assertLessEqual(d["l2_output_error"],
                                         d["error_bound_2B_epsilon"]+2e-12)

    def test_token_limit_and_shapes(self):
        with self.assertRaises(ValueError):
            dense_attention([0.]*1025,[[0.]]*1025)
        with self.assertRaises(ValueError):
            dense_attention([0.,1.],[[1.],[1.,2.]])
        with self.assertRaises(ValueError):
            attention_logits([1.],[[1.,2.]])


if __name__ == "__main__":
    unittest.main()
