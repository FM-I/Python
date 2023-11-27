from docx import Document
import random
import math
import matplotlib.pyplot as plt

# Зчитування даних з файлу Word
doc = Document('data.docx')

# Ініціалізація порожнього словника для збереження даних
cities_data = {}
skip_rows = 2
# Парсинг даних та зберігання їх у словнику
for table in doc.tables:
    for row_idx, row in enumerate(table.rows):
        # Пропустити перші два рядки
        if row_idx < skip_rows:
            continue
        # Отримання значень рядка таблиці
        values = [cell.text.strip() for cell in row.cells]
        # Перший елемент списку - номер, інші елементи - дані міста
        city_data = [int(value) if value.isdigit() else value for value in values[1:]]
        # Додавання даних до словника, де ключ - номер, значення - дані міста у вигляді списку
        cities_data[int(values[0])] = city_data

# Виведення отриманих даних
print(cities_data)


NUM_AGENTS = 10
EVAPORATION_RATE = 0.5
NUM_ITERATIONS = 100
ALPHA = 1.0  # Вага феромонів
BETA = 2.0   # Вага видимості (зворотної відстані)

class Ant:
    def __init__(self):
        self.visited_cities = set()
        self.travel_path = []
        self.total_distance = 0

def distance(city1, city2):
    if isinstance(city1, str) and city1.isdigit() and isinstance(city2, str) and city2.isdigit():
        return int(city1)  # Повертаємо вагу ребра між містами
    else:
        return float('inf')  # Невідома відстань

def calculate_path_distance(path):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance(path[i], path[i + 1])
    return total_distance

def initialize_agents():
    return [Ant() for _ in range(NUM_AGENTS)]

def choose_next_city(agent, pheromone_matrix, visibility_matrix):
    if not agent.travel_path:  # Якщо travel_path порожній, вибрати будь-яке місто
        unvisited_cities = list(cities_data.keys())
        current_city = random.choice(unvisited_cities)
    else:
        current_city = agent.travel_path[-1]
        unvisited_cities = set(cities_data.keys()) - agent.visited_cities

    if not unvisited_cities:  # Якщо всі міста вже відвідані, повернути будь-яке місто
        return random.choice(list(cities_data.keys()))

    probabilities = []
    for city in unvisited_cities:
        pheromone = pheromone_matrix[current_city - 1][city - 1] ** ALPHA
        visibility = 1 / distance(cities_data[current_city][city], cities_data[city][current_city]) ** BETA
        probabilities.append((city, pheromone * visibility))

    total_prob = sum(prob for _, prob in probabilities)
    if total_prob == 0:
        next_city = random.choice(list(unvisited_cities))
    else:
        probabilities = [(city, prob / total_prob) for city, prob in probabilities]
        next_city = random.choices(range(len(probabilities)), weights=[prob for _, prob in probabilities])[0]

    next_city_index = next_city % len(unvisited_cities)  # Виправлення індексу
    return list(unvisited_cities)[next_city_index]

def update_pheromones(pheromone_matrix, ants):
    for i in range(len(pheromone_matrix)):
        for j in range(len(pheromone_matrix[i])):
            pheromone_matrix[i][j] *= EVAPORATION_RATE
    for ant in ants:
        for i in range(len(ant.travel_path) - 1):
            if i + 1 < len(ant.travel_path):
                pheromone_matrix[ant.travel_path[i]][ant.travel_path[i + 1]] += 1 / ant.total_distance


def ant_colony_optimization():
    pheromone_matrix = [[1.0] * len(cities_data) for _ in range(len(cities_data))]
    best_path = None
    best_distance = float('inf')

    for _ in range(NUM_ITERATIONS):
        ants = initialize_agents()
        for ant in ants:
            while len(ant.visited_cities) < len(cities_data):
                next_city = choose_next_city(ant, pheromone_matrix, cities_data)
                ant.visited_cities.add(next_city)
                ant.travel_path.append(next_city)
            ant.total_distance = calculate_path_distance(ant.travel_path)
            if ant.total_distance < best_distance:
                best_distance = ant.total_distance
                best_path = ant.travel_path.copy()
        update_pheromones(pheromone_matrix, ants)

    print("Найкращий шлях:", best_path)
    print("Довжина найкращого шляху:", best_distance)

# Запускаємо метод мурашиних колоній
ant_colony_optimization()