"""
Класс автомобиля для симуляции самообучения с NEAT.
Реализует физику, зрение (raycasting) и обнаружение столкновений.
"""

import pygame
import math
import numpy as np


class Car:
    """Класс автомобиля с нейросетью и датчиками."""

    def __init__(self, x, y, angle=0, car_length=40, car_width=20):
        """
        Инициализация автомобиля.

        Args:
            x, y: начальные координаты центра автомобиля
            angle: начальный угол поворота (в градусах)
            car_length: длина автомобиля
            car_width: ширина автомобиля
        """
        # Позиция и ориентация
        self.x = x
        self.y = y
        self.angle = angle  # в градусах
        self.car_length = car_length
        self.car_width = car_width

        # Физические параметры
        self.speed = 0.0
        self.max_speed = 8.0
        self.acceleration = 0.15
        self.braking = 0.2
        self.friction = 0.05
        self.turn_speed = 4.0  # градусов за кадр
        self.turn_speed_at_high_speed = 2.0  # меньше поворот на высокой скорости

        # Состояние
        self.alive = True
        self.distance_traveled = 0.0
        self.time_alive = 0.0

        # Датчики (лучи)
        self.num_sensors = 7
        self.sensor_angles = [-90, -45, -20, 0, 20, 45, 90]  # относительные углы
        self.sensor_length = 150
        self.sensor_readings = [1.0] * self.num_sensors  # нормализованные расстояния

        # Создание маски для коллизий
        self._create_car_mask()

    def _create_car_mask(self):
        """Создает маску автомобиля для точного определения столкновений."""
        # Создаем поверхность для машины с прозрачностью
        car_surface = pygame.Surface((self.car_length, self.car_width), pygame.SRCALPHA)
        car_surface.fill((0, 0, 0, 0))  # прозрачный фон

        # Рисуем прямоугольник автомобиля
        car_surface.fill((255, 0, 0, 255), (0, 0, self.car_length, self.car_width))

        # Создаем маску из поверхности
        self.mask = pygame.mask.from_surface(car_surface)
        self.mask_surface = car_surface

    def get_corners(self):
        """
        Возвращает координаты углов автомобиля с учетом поворота.
        Используется для отрисовки и коллизий.
        """
        # Углы относительно центра
        half_length = self.car_length / 2
        half_width = self.car_width / 2

        # Углы в радианах
        rad_angle = math.radians(self.angle)

        # Углы автомобиля (до поворота)
        corners = [
            (-half_length, -half_width),  # левый передний
            (half_length, -half_width),   # правый передний
            (half_length, half_width),    # правый задний
            (-half_length, half_width)    # левый задний
        ]

        # Поворачиваем и перемещаем углы
        rotated_corners = []
        for cx, cy in corners:
            # Поворот
            rx = cx * math.cos(rad_angle) - cy * math.sin(rad_angle)
            ry = cx * math.sin(rad_angle) + cy * math.cos(rad_angle)
            # Перемещение
            rotated_corners.append((self.x + rx, self.y + ry))

        return rotated_corners

    def cast_ray(self, track_mask, start_pos, angle_deg):
        """
        Бросает луч и возвращает нормализованное расстояние до стены.

        Args:
            track_mask: маска трассы (1 = трасса, 0 = стена/фон)
            start_pos: начальная точка луча (x, y)
            angle_deg: угол луча в градусах

        Returns:
            Нормализованное расстояние (0.0 - 1.0), где 1.0 = луч не попал в стену
        """
        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        # Шаг проверки (в пикселях)
        step = 2
        max_steps = int(self.sensor_length / step)

        for step_i in range(1, max_steps + 1):
            check_x = int(start_pos[0] + dx * step_i * step)
            check_y = int(start_pos[1] + dy * step_i * step)

            # Проверяем границы экрана
            if check_x < 0 or check_x >= track_mask.get_size()[0] or \
               check_y < 0 or check_y >= track_mask.get_size()[1]:
                return step_i / max_steps  # вышли за границы

            # Проверяем маску трассы
            try:
                # Маска возвращает 1 если пиксель непрозрачный (трасса), 0 если прозрачный (стена)
                if track_mask.get_at((check_x, check_y)) == 0:  # стена (прозрачный пиксель)
                    return step_i / max_steps
            except IndexError:
                return step_i / max_steps

        return 1.0  # луч прошел всю дистанцию без препятствий

    def update_sensors(self, track_mask):
        """
        Обновляет показания всех датчиков.
        """
        # Центр автомобиля
        center = (int(self.x), int(self.y))

        # Обновляем каждый датчик
        for i, rel_angle in enumerate(self.sensor_angles):
            sensor_angle = self.angle + rel_angle
            self.sensor_readings[i] = self.cast_ray(track_mask, center, sensor_angle)

    def check_collision(self, track_mask):
        """
        Проверяет столкновение автомобиля с границами трассы.

        Args:
            track_mask: маска трассы

        Returns:
            True если есть столкновение, иначе False
        """
        # Получаем углы автомобиля
        corners = self.get_corners()
        
        # Добавляем середины сторон для более точной проверки
        mid_points = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i+1) % 4]
            mid_points.append(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))
        
        # Все точки для проверки: углы, середины и центр
        all_points = corners + mid_points + [(self.x, self.y)]
        
        # Проверяем каждую точку
        for point in all_points:
            cx, cy = int(point[0]), int(point[1])
            
            # Проверяем границы экрана
            if cx < 0 or cx >= track_mask.get_size()[0] or \
               cy < 0 or cy >= track_mask.get_size()[1]:
                return True
            
            # Проверяем маску трассы
            try:
                # Маска возвращает 1 если пиксель непрозрачный (трасса), 0 если прозрачный (стена)
                if track_mask.get_at((cx, cy)) == 0:  # стена (прозрачный пиксель)
                    return True
            except IndexError:
                return True
        
        return False

    def get_inputs(self):
        """
        Возвращает входные данные для нейросети.

        Returns:
            Список входных значений: [датчики..., скорость, угол поворота]
        """
        inputs = self.sensor_readings.copy()
        inputs.append(self.speed / self.max_speed)  # нормализованная скорость

        # Нормализованный угол поворота (-1 до 1)
        normalized_angle = (self.angle % 360) / 180.0
        if normalized_angle > 1.0:
            normalized_angle = 2.0 - normalized_angle
        inputs.append(normalized_angle)

        return inputs

    def control(self, outputs):
        """
        Управляет автомобилем на основе выходов нейросети.

        Args:
            outputs: список [газ, поворот_влево, поворот_вправо]
        """
        # outputs: [gas, left, right] в диапазоне 0-1

        # Газ
        if outputs[0] > 0.5:
            self.speed += self.acceleration
        elif outputs[0] < 0.3:
            self.speed -= self.braking

        # Торможение из-за трения
        if abs(self.speed) > self.friction:
            self.speed -= math.copysign(self.friction, self.speed)
        else:
            self.speed = 0.0

        # Ограничение скорости
        self.speed = max(-self.max_speed / 2, min(self.speed, self.max_speed))

        # Поворот (только если есть скорость)
        if abs(self.speed) > 0.1:
            turn_multiplier = 1.0 if abs(self.speed) < self.max_speed / 2 else 0.5
            if outputs[1] > outputs[2]:  # поворот влево
                self.angle -= self.turn_speed * turn_multiplier
            elif outputs[2] > outputs[1]:  # поворот вправо
                self.angle += self.turn_speed * turn_multiplier

        # Нормализуем угол
        self.angle %= 360

    def update(self, track_mask, dt=1.0):
        """
        Обновляет состояние автомобиля.

        Args:
            track_mask: маска трассы
            dt: дельта времени (для плавности)

        Returns:
            True если автомобиль жив, иначе False
        """
        if not self.alive:
            return False

        # Обновляем время жизни
        self.time_alive += dt

        # Двигаем автомобиль
        rad_angle = math.radians(self.angle)
        self.x += math.cos(rad_angle) * self.speed
        self.y += math.sin(rad_angle) * self.speed

        # Увеличиваем пройденное расстояние
        self.distance_traveled += abs(self.speed)

        # Обновляем датчики
        self.update_sensors(track_mask)

        # Проверяем коллизию
        if self.check_collision(track_mask):
            self.alive = False
            return False

        return True

    def draw(self, screen, is_leader=False):
        """
        Отрисовывает автомобиль на экране.

        Args:
            screen: поверхность pygame для отрисовки
            is_leader: является ли этот автомобиль лидером
        """
        # Получаем углы автомобиля
        corners = self.get_corners()

        # Цвет: лидер - зеленый, остальные - красные
        color = (0, 255, 0) if is_leader else (255, 0, 0)

        # Рисуем автомобиль как полигон
        pygame.draw.polygon(screen, color, corners, 2)

        # Рисуем направление (нос автомобиля)
        rad_angle = math.radians(self.angle)
        nose_x = self.x + math.cos(rad_angle) * (self.car_length / 2)
        nose_y = self.y + math.sin(rad_angle) * (self.car_length / 2)
        pygame.draw.circle(screen, (255, 255, 0), (int(nose_x), int(nose_y)), 3)

    def draw_sensors(self, screen):
        """
        Отрисовывает датчики (лучи) на экране.
        """
        center = (int(self.x), int(self.y))
        rad_angle = math.radians(self.angle)

        for i, rel_angle in enumerate(self.sensor_angles):
            sensor_angle = self.angle + rel_angle
            angle_rad = math.radians(sensor_angle)

            # Длина луча в зависимости от показания
            length = self.sensor_readings[i] * self.sensor_length

            end_x = self.x + math.cos(angle_rad) * length
            end_y = self.y + math.sin(angle_rad) * length

            # Цвет в зависимости от расстояния
            if self.sensor_readings[i] < 0.3:
                color = (255, 0, 0)  # красный - близко
            elif self.sensor_readings[i] < 0.6:
                color = (255, 255, 0)  # желтый - среднее
            else:
                color = (0, 255, 0)  # зеленый - далеко

            pygame.draw.line(screen, color, center, (end_x, end_y), 1)

    def calculate_fitness(self):
        """
        Вычисляет приспособленность (fitness) автомобиля.

        Returns:
            Значение fitness
        """
        # Основные компоненты:
        # 1. Пройденное расстояние
        distance_score = self.distance_traveled

        # 2. Время жизни
        time_score = self.time_alive * 10

        # 3. Бонус за скорость
        speed_bonus = max(0, self.speed) * 5

        # Если умер - штраф
        if not self.alive:
            death_penalty = -1000
        else:
            death_penalty = 0

        return distance_score + time_score + speed_bonus + death_penalty