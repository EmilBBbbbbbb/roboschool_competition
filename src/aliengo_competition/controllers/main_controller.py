from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from aliengo_competition.common.run_logger import CompetitionRunLogger
from aliengo_competition.robot_interface.base import AliengoRobotInterface
from aliengo_competition.robot_interface.types import CameraState


def _unwrap_env_from_robot(robot: AliengoRobotInterface):
    env = getattr(robot, "env", None)
    while env is not None and hasattr(env, "env") and getattr(env, "env") is not env:
        env = env.env
    return env


def _infer_control_dt(robot: AliengoRobotInterface, fallback_dt: float = 0.02) -> float:
    env = _unwrap_env_from_robot(robot)
    dt = getattr(env, "dt", None) if env is not None else None
    try:
        dt_value = float(dt)
        if dt_value > 0.0:
            return dt_value
    except (TypeError, ValueError):
        pass
    return float(fallback_dt)


class _CameraRenderer:
    def __init__(self, enabled: bool, depth_max_m: float):
        self.enabled = bool(enabled)
        self.depth_max_m = max(float(depth_max_m), 0.1)
        self._window_name = "Front Camera (Intel RealSense D435-like)"
        self._cv2 = None
        self._active = False
        if not self.enabled:
            return
        try:
            import cv2
        except Exception as exc:
            print(f"Отрисовка камеры отключена: не удалось импортировать cv2 ({exc})")
            self.enabled = False
            return
        self._cv2 = cv2
        self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_NORMAL)
        self._active = True

    def show(self, camera: CameraState) -> None:
        if not self._active or not isinstance(camera, CameraState):
            return
        image = camera.rgb
        depth = camera.depth
        if image is None or depth is None:
            return

        rgb = np.asarray(image)
        depth_m = np.asarray(depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or depth_m.ndim != 2:
            return
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = rgb[..., :3]
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=self.depth_max_m, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, self.depth_max_m)
        depth_u8 = (depth_m * (255.0 / self.depth_max_m)).astype(np.uint8)

        cv2 = self._cv2
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        view = np.concatenate((rgb_bgr, depth_color), axis=1)

        cv2.putText(view, "RGB", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            view,
            f"Depth 0..{self.depth_max_m:.1f}m",
            (rgb.shape[1] + 10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self._window_name, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()

    def close(self) -> None:
        if not self._active or self._cv2 is None:
            return
        self._cv2.destroyWindow(self._window_name)
        self._active = False


def run(
    robot: AliengoRobotInterface,
    steps: int = 15000,
    render_camera: bool = False,
    camera_depth_max_m: float = 4.0,
    seed: int = 0,
) -> None:
    robot.reset()
    env = getattr(robot, "env", None)
    if env is None:
        raise ValueError("Интерфейс робота должен предоставлять 'env' для обязательного логирования.")

    logger = CompetitionRunLogger(env=env, seed=int(seed))
    camera_renderer = _CameraRenderer(enabled=render_camera, depth_max_m=camera_depth_max_m)
    control_dt = _infer_control_dt(robot, fallback_dt=0.02)
    requested_steps = max(int(steps), 1)
    nominal_dt = 0.02
    target_duration_s = requested_steps * nominal_dt
    total_steps = max(int(round(target_duration_s / control_dt)), 1)
    print(
        f"[Контроллер] dt={control_dt:.4f}с, requested_steps={requested_steps}, "
        f"effective_steps={total_steps}"
    )
    object_queue = list(getattr(env, "SEQUENCE_OF_OBJECTS", []))
    print(f"[Контроллер] отрисовка_камеры={'включена' if camera_renderer.enabled else 'выключена'}")
    print(f"[Контроллер] object_queue={object_queue}")

    # Редактируемые пользователем блоки в этом файле:
    # 1. USER PARAMETERS START / END
    # 2. USER CONTROL LOGIC START / END

    # ================= USER PARAMETERS START =================
    # Настраивайте эти значения, чтобы менять поведение демо.
    # Параметры, завязанные на время, пересчитываются через шаг симуляции, потому время в секундах работают в симуляции правильно
    warmup_s = 0.4
    ramp_s = 1.2
    trajectory_period_s = 8.0
    forward_speed_mean = 0.40
    forward_speed_amp = 0.35
    lateral_speed_amp = 0.22
    yaw_rate_amp = 0.75
    yaw_rate_damping = 0.55
    ang_vel_scale = 0.25

    # Параметры детекции объектов через YOLO.
    enable_yolo_detection = True
    yolo_conf_threshold = 0.55
    yolo_imgsz = 416
    yolo_detect_every_n_steps = 3
    yolo_min_box_area_ratio = 0.01
    yolo_min_streak = 2
    yolo_device = "cuda:0"
    yolo_weights_override: str | None = None

    repo_root = Path(__file__).resolve().parents[3]
    yolo_weights_path = (
        Path(yolo_weights_override).expanduser().resolve()
        if yolo_weights_override
        else repo_root / "weights" / "last.pt"
    )

    yolo_model = None
    yolo_enabled = False
    yolo_step_counter = 0
    yolo_last_candidate_id: int | None = None
    yolo_candidate_streak = 0
    # ================== USER PARAMETERS END ==================

    segment_start_t = 0.0

    try:
        initial_observation = robot.get_observation()
        initial_camera_payload = robot.get_camera()
        print(
            "[Контроллер] Предпросмотр API:"
            f" observation_type={type(initial_observation).__name__},"
            f" camera_payload={'да' if initial_camera_payload is not None else 'нет'}"
        )
        if initial_camera_payload is None:
            print(
                "[Контроллер] Предупреждение: данные фронтальной камеры недоступны. "
                "Проверьте, что симулятор не запущен в headless-режиме и что включён front_camera_enabled."
            )

        object_name_to_id = {}
        queue_ids = set()
        for item in object_queue:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                object_id = int(item[0])
                object_name = str(item[1]).strip().lower()
                object_name_to_id[object_name] = object_id
                queue_ids.add(object_id)

        if enable_yolo_detection:
            try:
                from ultralytics import YOLO

                if not yolo_weights_path.exists():
                    raise FileNotFoundError(f"YOLO weights not found: {yolo_weights_path}")

                yolo_model = YOLO(str(yolo_weights_path))
                yolo_enabled = True
                print(f"[Контроллер] YOLO детектор загружен: {yolo_weights_path}")
                print(f"[Контроллер] YOLO device: {yolo_device}")
                print(f"[Контроллер] map name->id: {object_name_to_id}")
                if yolo_device.startswith("cuda") and not torch.cuda.is_available():
                    yolo_enabled = False
                    print("[Контроллер] YOLO отключен: CUDA недоступна в текущем окружении.")
            except Exception as exc:
                yolo_enabled = False
                print(f"[Контроллер] YOLO отключен: {exc}")

        for step_index in range(total_steps):
            state = robot.get_state()

            # Камеру можно брать и из state, и напрямую через robot.get_camera().
            camera_payload = robot.get_camera()
            camera_state = state.camera
            if (camera_state.rgb is None or camera_state.depth is None) and isinstance(camera_payload, dict):
                camera_state = CameraState(
                    rgb=camera_payload.get("image"),
                    depth=camera_payload.get("depth"),
                )
            elif (camera_state.rgb is None or camera_state.depth is None) and isinstance(camera_payload, CameraState):
                camera_state = camera_payload
            camera_renderer.show(camera_state)
            omega_z = state.imu.wz / ang_vel_scale

            # ================= USER CONTROL LOGIC START =================
            # Это основной блок для логики участника.
            # Здесь нужно читать измерения, принимать решение и формировать
            # команды движения. Логирование найденного объекта тоже делается
            # отсюда.
            #
            # Формат данных эквивалентен данных:
            # - вход команды: vx, vy, wz
            # - выход состояния: measured_vx, measured_vy, measured_wz
            # - joint_states: joint_names, relative_dof_pos, dof_vel
            # - imu: base_ang_vel, base_lin_acc
            # - camera: camera_data["image"], camera_data["depth"]
            # - порядок объектов: object_queue
            #
            # Ниже приведён обязательный шаблон. Участник должен:
            # 1. реализовать get_found_object_id(...)
            # 2. при обнаружении объекта вернуть его id
            # 3. обязательно вызвать log_found_object(...)
            #
            # Если объект не найден, верните None.
            sim_t = state.sim_time_s

            joint_names = state.joints.name
            relative_dof_pos = state.q
            dof_vel = state.q_dot
            measured_vx = state.vx
            measured_vy = state.vy
            measured_wz = state.wz
            base_ang_vel = state.imu.angular_velocity_xyz
            base_lin_acc = np.zeros(3, dtype=np.float32)
            camera_data = camera_payload if isinstance(camera_payload, dict) else {
                "image": camera_state.rgb,
                "depth": camera_state.depth,
            }

            # Обязательная обвязка для логирования найденного объекта.
            # Использование:
            # - get_found_object_id(...) должен вернуть id найденного объекта
            #   или None, если объект не найден
            # - log_found_object(...) записывает событие в судейский лог
            # - этот шаблон нельзя удалять: участник обязан реализовать его
            #   в своём решении
            #
            # Пример:
            #     detected_object_id = get_found_object_id(...)
            #     if detected_object_id is not None:
            #         log_found_object(detected_object_id)
            def log_found_object(object_id: int) -> None:
                """ОБЯЗАТЕЛЬНО: вызывайте при обнаружении целевого объекта."""
                logger.log_detected_object_at_time(int(object_id), float(sim_t))

            def get_found_object_id(
                current_state,
                current_camera_data,
                current_object_queue,
            ):
                """Возвращает id обнаруженного объекта из object_queue или None."""
                nonlocal yolo_step_counter, yolo_last_candidate_id, yolo_candidate_streak

                if not yolo_enabled or yolo_model is None:
                    return None

                yolo_step_counter += 1
                if yolo_step_counter % max(int(yolo_detect_every_n_steps), 1) != 0:
                    return None

                if not isinstance(current_camera_data, dict):
                    return None
                image = current_camera_data.get("image")
                if image is None:
                    return None

                rgb = np.asarray(image)
                if rgb.ndim != 3 or rgb.shape[2] < 3:
                    return None
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                rgb = rgb[..., :3]

                # Ultralytics корректно принимает ndarray в формате BGR.
                bgr = rgb[..., ::-1]

                try:
                    results = yolo_model.predict(
                        source=bgr,
                        conf=float(yolo_conf_threshold),
                        imgsz=int(yolo_imgsz),
                        device=yolo_device,
                        verbose=False,
                    )
                except Exception:
                    return None

                if not results:
                    return None

                result = results[0]
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    return None

                try:
                    confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
                    classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
                    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                except Exception:
                    return None

                if confs.size == 0:
                    return None
                best_idx = int(np.argmax(confs))

                image_area = float(rgb.shape[0] * rgb.shape[1])
                if image_area <= 0.0:
                    return None
                box = xyxy[best_idx]
                box_area = max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
                if (box_area / image_area) < float(yolo_min_box_area_ratio):
                    return None

                best_class = int(classes[best_idx])
                names = getattr(result, "names", {})
                predicted_name = str(names.get(best_class, best_class)).strip().lower()

                detected_id = object_name_to_id.get(predicted_name)
                if detected_id is None and best_class in queue_ids:
                    detected_id = best_class
                if detected_id is None:
                    return None

                if yolo_last_candidate_id == detected_id:
                    yolo_candidate_streak += 1
                else:
                    yolo_last_candidate_id = detected_id
                    yolo_candidate_streak = 1

                if yolo_candidate_streak < max(int(yolo_min_streak), 1):
                    return None
                return int(detected_id)

            detected_object_id = get_found_object_id(
                state,
                camera_data,
                object_queue,
            )
            if detected_object_id is not None:
                log_found_object(detected_object_id)

            local_t = max(sim_t - segment_start_t, 0.0)
            if local_t < warmup_s:
                vx = 0.0
                vy = 0.0
                vw = 0.0
            else:
                motion_t = local_t - warmup_s
                phase = 2.0 * math.pi * motion_t / max(trajectory_period_s, control_dt)
                ramp = min(motion_t / max(ramp_s, control_dt), 1.0)

                vx = ramp * (forward_speed_mean + forward_speed_amp * math.cos(phase))
                vy = ramp * (lateral_speed_amp * math.sin(2.0 * phase))
                yaw_ff = yaw_rate_amp * math.sin(phase + math.pi / 4.0)
                vw = ramp * (yaw_ff - yaw_rate_damping * state.imu.wz / ang_vel_scale)
                vw = max(min(vw, 1.0), -1.0)
            # ================== USER CONTROL LOGIC END ==================

            robot.set_speed(vx, vy, vw)
            robot.step()
            logger.log_step(step_index * control_dt)
            robot.get_observation()  # Пример доступа к наблюдению после step().

            if robot.is_fallen():
                robot.stop()
                robot.reset()
                segment_start_t = (step_index + 1) * control_dt
                print("[Контроллер] робот упал -> сброс")
                continue
    finally:
        logger.close()
        camera_renderer.close()
        robot.stop()
