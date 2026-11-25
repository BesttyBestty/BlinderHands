import math

import numpy as np
import pytest

from blinders_hands.arm import ArmError, MockArm
from blinders_hands.grasp import (GraspConfig, estimate_object, execute_grasp, place_back, plan_grasp,
                                  solve_rigid_transform, synthetic_scene, transform_points)


def test_kabsch_recovers_transform():
    rng = np.random.default_rng(0)
    a = 0.7
    T = np.eye(4)
    T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = [0.3, -0.1, 0.8]
    src = rng.uniform(-0.3, 0.3, (8, 3))
    est = solve_rigid_transform(src, transform_points(T, src))
    assert np.allclose(est, T, atol=1e-9)


def test_estimate_box_pose():
    frame, T, bbox = synthetic_scene([(0.40, 0.10, 0.09, 0.045, 0.06, 0.5)])
    e = estimate_object(frame, bbox[0], T)
    assert np.allclose(e.center_base, [0.40, 0.10, 0.06], atol=0.005)
    assert abs(e.length - 0.09) < 0.006 and abs(e.width - 0.045) < 0.006 and abs(e.height - 0.06) < 0.003
    # 长方体关于长边对称：朝向只需在 π 范围内对上
    assert abs(math.remainder(e.yaw_base - 0.5, math.pi)) < 0.05


def test_plan_and_execute_grasp():
    frame, T, bbox = synthetic_scene()
    cfg = GraspConfig()
    plan = plan_grasp(estimate_object(frame, bbox, T), cfg)
    assert abs(plan.grasp[5] - 0.5) < 0.05                   # 夹爪顺着长边
    assert plan.pregrasp[2] - plan.grasp[2] == pytest.approx(cfg.approach_height)
    assert plan.object_width < plan.open_width <= cfg.gripper_max_width
    arm = MockArm()
    execute_grasp(arm, plan, cfg)
    kinds = [h[0] for h in arm.history]
    assert kinds == ["gripper", "move_p", "move_l", "gripper", "move_l"]
    assert arm.history[3][1] == 0.0                           # 夹紧
    place_back(arm, plan)
    assert arm.history[-2] == ("gripper", round(plan.open_width, 4))


def test_rejects_too_wide_and_out_of_reach():
    frame, T, bbox = synthetic_scene([(0.40, 0.0, 0.12, 0.10, 0.05, 0.0)])
    with pytest.raises(ArmError, match="太宽"):
        plan_grasp(estimate_object(frame, bbox[0], T), GraspConfig())
    frame, T, bbox = synthetic_scene()
    cfg = GraspConfig(workspace=[[0.5, 0.6], [-0.1, 0.1], [0.0, 0.5]])
    with pytest.raises(ArmError, match="超出安全工作空间"):
        plan_grasp(estimate_object(frame, bbox, T), cfg)
