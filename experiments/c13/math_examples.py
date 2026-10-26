"""C13: float64 mathematical examples; no model/data downloads."""
import json
from pathlib import Path
import numpy as np

def examples():
    a = np.array([[2., 1.], [0., .5]])
    x = np.array([1., -2.])
    y = a @ x
    q, _ = np.linalg.qr(np.array([[1., 2.], [3., 1.]]))
    u = np.array([[1.,0.],[0.,1.],[0.,0.]])
    restricted = np.array([[2.,1.,9.],[0.,3.,-4.]])
    xs = u @ np.array([1., -2.])
    ys = restricted @ xs
    recovery = u @ np.linalg.pinv(restricted @ u) @ ys
    p = np.array([[1.,0.,0.],[0.,1.,0.]])
    x1 = np.array([1.,2.,3.]); x2 = np.array([1.,2.,-7.])
    eps = 1e-5
    v = np.array([1.,2.,4.])
    gamma = np.array([2.,.5,1.5]); beta = np.array([.2,-.1,.7])
    def ln(t):
        return gamma*(t-t.mean())/np.sqrt(t.var()+eps)+beta
    z = ln(v)
    restored = ((z-beta)/gamma)*np.sqrt(v.var()+eps)+v.mean()
    def attention(t):
        scores = t @ t.T / np.sqrt(t.shape[1])
        probs = np.exp(scores-scores.max(axis=1,keepdims=True))
        probs /= probs.sum(axis=1,keepdims=True)
        return probs @ t, probs
    tokens = np.array([[1.,.2],[-.3,.7],[.5,-.8]])
    out, probs = attention(tokens)
    trans_back = probs.T @ out
    fixed_pinv = np.linalg.pinv(probs) @ out
    # Input-dependent uniform attention: Q=K=0, V=identity.
    uniform = np.ones((2,2))/2
    c1 = np.array([[1.,0.],[-1.,0.]])
    c2 = np.array([[0.,2.],[0.,-2.]])
    # Autodiff/VJP of a squared-error objective is an adjoint, not an inverse.
    grad_at_zero = a.T @ (a @ np.zeros(2)-y)
    # A one-dimensional image family remains exactly reconstructible through
    # dimensional reduction, provided its restricted map is injective.
    template = np.arange(16,dtype=float)+1
    measurement = np.ones((1,16))
    image = .37*template
    recovered_image = template*((measurement@image).item()/(measurement@template).item())
    r = {
      "full_rank_square": {
        "A":a.tolist(),"x":x.tolist(),"y":y.tolist(),
        "transpose_error":float(np.linalg.norm(a.T@y-x)),
        "pinverse_error":float(np.linalg.norm(np.linalg.pinv(a)@y-x)),
        "orthogonal_transpose_error":float(np.linalg.norm(q.T@(q@x)-x)),
        "condition_number":float(np.linalg.cond(a)),
        "backprop_gradient_at_zero":grad_at_zero.tolist(),
        "gradient_is_inverse":False},
      "known_subspace": {
        "A":restricted.tolist(),"U":u.tolist(),"x":xs.tolist(),
        "restricted_rank":int(np.linalg.matrix_rank(restricted@u)),
        "ambient_pinverse_error":float(np.linalg.norm(np.linalg.pinv(restricted)@ys-xs)),
        "known_subspace_recovery_error":float(np.linalg.norm(recovery-xs))},
      "dimension_reduction": {
        "A":p.tolist(),"x1":x1.tolist(),"x2":x2.tolist(),
        "same_output":bool(np.array_equal(p@x1,p@x2)),
        "pinverse_x1":(np.linalg.pinv(p)@(p@x1)).tolist(),
        "one_dimensional_image_family_recovery_error":float(np.linalg.norm(recovered_image-image))},
      "layer_norm": {
        "eps":eps,"gamma":gamma.tolist(),"beta":beta.tolist(),
        "x1":v.tolist(),"x2":(v+10).tolist(),
        "shift_collision_max_error":float(np.max(np.abs(ln(v)-ln(v+10)))),
        "scale_difference_with_nonzero_eps":float(np.max(np.abs(ln(v)-ln(3*v)))),
        "restore_with_mean_variance_sidechannel_error":float(np.linalg.norm(restored-v)),
        "sidechannel":{"mean":float(v.mean()),"variance":float(v.var())},
        "mean_variance_are_extra_information":True},
      "attention": {
        "definition":"Q=X, K=X, V=X; row-softmax(XX^T/sqrt(d))X",
        "input":tokens.tolist(),"P":probs.tolist(),
        "transpose_reconstruction_error":float(np.linalg.norm(trans_back-tokens)),
        "pinverse_error_if_original_P_is_supplied":float(np.linalg.norm(fixed_pinv-tokens)),
        "recompute_P_from_output_error":float(np.linalg.norm(attention(out)[0]-tokens)),
        "original_P_is_input_dependent_side_information":True,
        "uniform_QK_zero_collision":{"x1":c1.tolist(),"x2":c2.tolist(),
          "outputs_equal":bool(np.array_equal(uniform@c1,uniform@c2))}},
      "invertible_nonlinearity":{
        "function":"tanh elementwise, real finite input; inverse arctanh for |y|<1",
        "inverse_error":float(np.linalg.norm(np.arctanh(np.tanh(v/10))-v/10)),
        "reapply_tanh_error":float(np.linalg.norm(np.tanh(np.tanh(v/10))-v/10))}
    }
    assert r["full_rank_square"]["pinverse_error"]<1e-12
    assert r["known_subspace"]["known_subspace_recovery_error"]<1e-12
    assert r["layer_norm"]["restore_with_mean_variance_sidechannel_error"]<1e-12
    assert r["dimension_reduction"]["same_output"]
    assert r["attention"]["uniform_QK_zero_collision"]["outputs_equal"]
    return r

if __name__=="__main__":
    print(json.dumps(examples(),ensure_ascii=False,indent=2))
