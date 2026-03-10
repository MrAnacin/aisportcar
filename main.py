"""
Основной файл симуляции самообучающегося гонщика с NEAT.
Содержит игровой цикл, эволюцию нейросетей и визуализацию.
"""

import pygame
import neat
import random
import math
import os
import numpy as np
from car import Car


class TrackGenerator:
    """Генератор трассы. Создает трассу программно."""

    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.track_surface = None
        self.track_mask = None

    def generate_oval_track(self):
        """
        Генерирует овальную трассу с помощью pygame.draw.
        Дорога - белая (альфа=255), стены - прозрачные (альфа=0).
        """
        # Создаем поверхность с альфа-каналом
        self.track_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.track_surface.fill((0, 0, 0, 0))  # прозрачный фон (стены)

        # Центр трассы
        cx, cy = self.width // 2, self.height // 2

        # Размеры овала
        rx = self.width * 0.4  # горизонтальный радиус
        ry = self.height * 0.35  # вертикальный радиус

        # Рисуем белую дорогу (заполненный овал)
        pygame.draw.ellipse(
            self.track_surface,
            (255, 255, 255, 255),  # непрозрачная белая трасса
            (cx - rx, cy - ry, rx * 2, ry * 2),
            0  # заполненный
        )

        # Создаем маску (белое = 1, прозрачное = 0)
        self.track_mask = pygame.mask.from_surface(self.track_surface)

        # Сохраняем трассу в файл
        pygame.image.save(self.track_surface, "track.png")
        print("Трасса сохранена как 'track.png'")

        return self.track_surface, self.track_mask

    def generate_winding_track(self):
        """
        Генерирует извилистую трассу с помощью полигонов.
        """
        self.track_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.track_surface.fill((0, 0, 0, 0))  # прозрачный фон

        # Создаем трассу из нескольких соединенных сегментов
        points = []
        center_x, center_y = self.width // 2, self.height // 2

        # Генерируем замкнутую кривую с шумом
        num_points = 20
        base_radius = min(self.width, self.height) * 0.35

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            # Добавляем случайное отклонение
            r = base_radius + random.uniform(-50, 50)
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            points.append((x, y))

        # Рисуем трассу как выпуклый многоугольник
        pygame.draw.polygon(self.track_surface, (255, 255, 255, 255), points, 0)

        # Сглаживаем, рисуя по несколько раз с разной толщиной
        for width_px in range(60, 30, -10):
            pygame.draw.polygon(self.track_surface, (255, 255, 255, 255), points, width_px)

        self.track_mask = pygame.mask.from_surface(self.track_surface)
        pygame.image.save(self.track_surface, "track.png")
        print("Трасса сохранена как 'track.png'")

        return self.track_surface, self.track_mask


class Simulation:
    """Основной класс симуляции с NEAT."""

    def __init__(self, config_path, track_mask=None, screen_width=1200, screen_height=800):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.track_mask = track_mask

        # Загрузка конфигурации NEAT
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )

        # Создание популяции
        self.population = None
        self.generation = 0

        # Список автомобилей
        self.cars = []
        self.best_car = None

        # Статистика
        self.fps_history = []

    def create_cars(self, genomes, config):
        """
        Создает автомобили из геномов.

        Args:
            genomes: список (genome_id, genome) из NEAT
            config: конфигурация NEAT
        """
        self.cars = []

        for genome_id, genome in genomes:
            # Создаем нейросеть из генома
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            # Стартовая позиция (внутри трассы)
            start_x = self.screen_width // 2
            start_y = self.screen_height // 2 + 100
            start_angle = -90  # смотрим вверх

            car = Car(start_x, start_y, angle=start_angle)
            car.neural_net = net
            car.genome = genome
            car.fitness = 0.0

            self.cars.append(car)

    def evaluate_genomes(self, genomes, config):
        """
        Функция оценки для NEAT. Запускает симуляцию для всех геномов.
        """
        self.generation += 1
        self.create_cars(genomes, config)

        # Основной цикл симуляции
        max_time = 1000  # максимальное время жизни в кадрах
        time = 0

        while time < max_time:
            # Проверяем, есть ли живые автомобили
            alive_cars = [car for car in self.cars if car.alive]
            if not alive_cars:
                break

            # Управляем каждым автомобилем
            for car in alive_cars:
                # Получаем входы от датчиков
                inputs = car.get_inputs()

                # Пропускаем через нейросеть
                outputs = car.neural_net.activate(inputs)

                # Применяем управление
                car.control(outputs)

                # Обновляем состояние
                car.update(self.track_mask)

            time += 1

        # Вычисляем fitness для каждого генома
        for i, (genome_id, genome) in enumerate(genomes):
            if i < len(self.cars):
                genome.fitness = self.cars[i].calculate_fitness()

        # Находим лучший автомобиль
        best_fitness = -1
        best_idx = 0
        for i, car in enumerate(self.cars):
            if car.fitness > best_fitness:
                best_fitness = car.fitness
                best_idx = i
        self.best_car = self.cars[best_idx]

    def run_neat(self, num_generations=50):
        """
        Запускает эволюцию NEAT.

        Args:
            num_generations: количество поколений
        """
        # Добавляем функцию оценки в конфиг
        self.config.genome_config.fitness_function = self.evaluate_genomes

        # Создаем популяцию
        self.population = neat.Population(self.config)

        # Добавляем статистику
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)

        # Запускаем эволюцию
        winner = self.population.run(self.evaluate_genomes, num_generations)

        print(f"Лучший геном: {winner}")
        return winner

    def visualize_best(self, screen, clock):
        """
        Визуализирует лучшего автомобиля после обучения.
        """
        if self.best_car is None:
            print("Нет лучшего автомобиля для визуализации!")
            return

        car = self.best_car
        car.alive = True
        car.speed = 0
        car.x = self.screen_width // 2
        car.y = self.screen_height // 2 + 100
        car.angle = -90

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Управляем лучшим автомобилем
            inputs = car.get_inputs()
            outputs = car.neural_net.activate(inputs)
            car.control(outputs)
            car.update(self.track_mask)

            # Отрисовываем
            screen.fill((100, 100, 100))  # серый фон
            screen.blit(self.track_surface, (0, 0))

            car.draw(screen, is_leader=True)
            car.draw_sensors(screen)

            # Статистика
            self.draw_stats(screen, [car], is_best_only=True)

            pygame.display.flip()
            clock.tick(60)

    def draw_stats(self, screen, cars, is_best_only=False):
        """
        Отрисовывает статистику на экране.
        """
        font = pygame.font.Font(None, 36)

        # Поколение
        gen_text = font.render(f"Поколение: {self.generation}", True, (255, 255, 255))
        screen.blit(gen_text, (10, 10))

        # Живые автомобили
        alive_count = sum(1 for car in cars if car.alive)
        alive_text = font.render(f"Живые: {alive_count}/{len(cars)}", True, (255, 255, 255))
        screen.blit(alive_text, (10, 50))

        # FPS
        fps = int(clock.get_fps())
        fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 90))

        if is_best_only and self.best_car:
            # Информация о лучшем
            best_font = pygame.font.Font(None, 28)
            dist_text = best_font.render(f"Дистанция: {self.best_car.distance_traveled:.1f}", True, (255, 255, 255))
            screen.blit(dist_text, (10, 130))

            speed_text = best_font.render(f"Скорость: {self.best_car.speed:.2f}", True, (255, 255, 255))
            screen.blit(speed_text, (10, 160))


def main():
    """Точка входа."""
    # Инициализация pygame
    pygame.init()

    # Размеры экрана
    WIDTH, HEIGHT = 1200, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Самообучающийся гонщик NEAT")
    clock = pygame.time.Clock()

    # Генератор трассы
    print("Генерация трассы...")
    track_gen = TrackGenerator(WIDTH, HEIGHT)

    # Проверяем, есть ли уже файл трассы
    if os.path.exists("track.png"):
        print("Загрузка существующей трассы...")
        track_surface = pygame.image.load("track.png").convert()
        track_mask = pygame.mask.from_surface(track_surface)
    else:
        # Генерируем новую трассу
        track_surface, track_mask = track_gen.generate_oval_track()
        # Или используем извилистую: track_surface, track_mask = track_gen.generate_winding_track()

    # Создание симуляции
    sim = Simulation("config-feedforward", track_mask, WIDTH, HEIGHT)

    # Режим работы
    print("\nВыберите режим:")
    print("1 - Обучить нейросеть (NEAT)")
    print("2 - Визуализировать лучшего (после обучения)")
    mode = input("Введите номер режима (1/2): ").strip()

    if mode == "1":
        # Обучение
        print("\nЗапуск обучения...")
        sim.run_neat(num_generations=30)

        # Сохраняем лучший геном
        if sim.best_car:
            with open("best_genome.pkl", "wb") as f:
                import pickle
                pickle.dump(sim.best_car.genome, f)
            print("Лучший геном сохранен в 'best_genome.pkl'")

        print("\nОбучение завершено!")
        print("Запустите программу снова и выберите режим 2 для визуализации.")

    elif mode == "2":
        # Загружаем лучший геном
        if os.path.exists("best_genome.pkl"):
            with open("best_genome.pkl", "rb") as f:
                import pickle
                best_genome = pickle.load(f)

            # Создаем сеть
            best_net = neat.nn.FeedForwardNetwork.create(best_genome, sim.config)

            # Создаем автомобиль
            start_x = WIDTH // 2
            start_y = HEIGHT // 2 + 100
            best_car = Car(start_x, start_y, angle=-90)
            best_car.neural_net = best_net
            best_car.genome = best_genome
            sim.best_car = best_car

            # Визуализация
            print("\nВизуализация лучшего автомобиля...")
            sim.visualize_best(screen, clock)
        else:
            print("Файл 'best_genome.pkl' не найден!")
            print("Сначала обучите нейросеть (режим 1).")
    else:
        print("Неверный режим!")

    pygame.quit()


if __name__ == "__main__":
    main()