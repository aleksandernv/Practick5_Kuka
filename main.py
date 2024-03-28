import time
import pygame
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
from KUKA import KUKA

def calc_coords10(coordinate1, k1):
    return (coordinate1 + k1) / 10

def calc_coords100(coordinate1, k1):
    return (coordinate1 + k1) / 100

def map_expansion(expansion, original_map):
    # Функция для расширения препятствий
    kernel_size = 2 * expansion + 1  # Размер ядра = 2 * 30 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_obstacles = cv2.dilate(original_map.astype(np.uint8), kernel)

    return expanded_obstacles

def paint_map(discr, sigma_blur, koef, expansion_size):
    with open('log.txt', 'r') as file:
        data_coords = []
        data_lens = []
        for line in file:
            parts = line.strip().split(";")
            parts_coords = parts[0].strip().split("\n")
            for i in parts_coords:
                coords = i.strip().split(", ")
                mas = []
                for j in coords:
                    mas.append(float(j))
                data_coords.append(mas)

            parts_len = parts[1].strip().split("\n")
            for i in parts_len:
                lengthes = i.strip().split(", ")
                mas = []
                for j in lengthes:
                    mas.append(float(j))
                data_lens.append(mas)

    # Обработка данных из  файла
    coords = []
    accuracy = 6
    for j in range(len(data_coords)):
        x_rob = data_coords[j][0]
        y_rob = data_coords[j][1]
        alpha_rob = data_coords[j][2]
        for k in range(len(data_lens[j])):
            alpha_lidar = k * (4 * math.pi / 3) / len(data_lens[j]) - 2 * math.pi / 3
            if data_lens[j][k] != 5.6 and alpha_lidar > -1.66 and alpha_lidar < 1.83:
                coords.append((round(data_lens[j][k] * math.cos(alpha_rob - alpha_lidar) + x_rob, accuracy),
                               round(data_lens[j][k] * math.sin(alpha_rob - alpha_lidar) + y_rob, accuracy)))

    x = []
    y = []

    for i in range(len(coords)):
        x.append(coords[i][0])
        y.append(coords[i][1])

    # Дискретизация
    min_x = int(min(x) * discr)
    min_y = int(min(y) * discr)

    original_map = np.zeros((12*discr, 12*discr))
    for i in range(len(x)):
        if abs(y[i]*discr - min_y) < len(original_map) and abs(x[i]*discr - min_x) < len(original_map):
            original_map[int(y[i] * discr) - min_y][int(x[i] * discr) - min_x] = 1

    blur = scipy.ndimage.filters.gaussian_filter(np.copy(original_map), sigma=sigma_blur)
    for i in range(len(blur)):
        for j in range(len(blur[i])):
            if blur[i][j] > koef:
                blur[i][j] = 1
            else:
                blur[i][j] = 0

    # Расширение препятствий
    expansion = expansion_size
    extended_map = map_expansion(expansion, np.copy(blur))

    return min_x, min_y, x_rob, y_rob, extended_map


#######################################################################
def evristick(map, start, end, evr):
  # Эвристическая оценка для точки

  x_s = start%len(map[0])
  y_s = start//len(map[0])

  # x_s, y_s = start
  x_e, y_e = end

  if evr == 0: # Манхетонская
    return (math.fabs(x_e-x_s) + math.fabs(y_e-y_s))
  if evr == 1: # Чебышева
    return max(math.fabs(x_e-x_s), math.fabs(y_e-y_s))
  else:
    return math.sqrt((x_e-x_s)**2 + (y_e-y_s)**2)

def evristick_line(map, line, end, evr):
  # Эвристическая оценка для строки
  new_line = np.zeros(len(line))
  line_0 = np.copy(line)
  for i in range(len(line)):
    line_0[i] = evristick(map, i, end, evr)

  return line_0

def iteration_of_astar(index, last_line, table):
  # Итерация Астар
  new_line = np.copy(last_line)
  new_line[index] = 0
  last_step = last_line[index]

  for i in range(len(new_line)):
    if table[index, i] + last_step < new_line[i]:
      new_line[i] = table[index, i] + last_step

  return new_line

def table_of_map_astar(map, evr):
  # Граф карты в виде таблицы
  weights = np.ones((len(map)*len(map[0]), (len(map)*len(map[0])))) * np.inf

  width = len(map[0])
  length = len(map)

  for i in range(len(map)):
    for j in range(len(map[i])):

      if map[i][j] != 1:

        weights[width * i + j][width * i + j] = 0

        if j != width - 1:
          if map[i][j+1] == 0: # правая точка
            weights[width * i + j][width * i + j + 1] = 1
          else:
            weights[width * i + j][width * i + j + 1] = np.inf

        if j != 0:
          if map[i][j-1] == 0: # левая точка
            weights[width * i + j][width * i + j - 1] = 1
          else:
            weights[width * i + j][width * i + j - 1] = np.inf

        if i != 0:
          if map[i-1][j] == 0: # верхняя точка
            weights[width * i + j][width * (i - 1) + j] = 1
          else:
            weights[width * i + j][width * (i - 1) + j] = np.inf

        if i != length -1:
          if map[i+1][j] == 0: # нижняя точка
            weights[width * i + j][width * (i + 1) + j] = 1
          else:
            weights[width * i + j][width * (i + 1) + j] = np.inf

        if evr == 1:

          if i != length -1 and j != width - 1:
            if map[i+1][j+1] == 0: # правая нижняя точка
              weights[width * i + j][width * (i + 1) + j + 1] = 1
            else:
              weights[width * i + j][width * (i + 1) + j + 1] = np.inf

          if i != 0 and j != width - 1:
            if map[i-1][j+1] == 0: # правая верхняя точка
              weights[width * i + j][width * (i - 1) + j + 1] = 1
            else:
              weights[width * i + j][width * (i - 1) + j + 1] = np.inf

          if i != length -1 and j != 0:
            if map[i+1][j-1] == 0: # левая нижняя точка
              weights[width * i + j][width * (i + 1) + j - 1] = 1
            else:
              weights[width * i + j][width * (i + 1) + j - 1] = np.inf

          if i != 0 and j != 0:
            if map[i-1][j-1] == 0: # левая верхняя точка
              weights[width * i + j][width * (i - 1) + j - 1] = 1
            else:
              weights[width * i + j][width * (i - 1) + j - 1] = np.inf

        if evr == 2:

          if i != length -1 and j != width - 1:
            if map[i+1][j+1] == 0: # правая нижняя точка
              weights[width * i + j][width * (i + 1) + j + 1] = 1.41
            else:
              weights[width * i + j][width * (i + 1) + j + 1] = np.inf

          if i != 0 and j != width - 1:
            if map[i-1][j+1] == 0: # правая верхняя точка
              weights[width * i + j][width * (i - 1) + j + 1] = 1.41
            else:
              weights[width * i + j][width * (i - 1) + j + 1] = np.inf

          if i != length -1 and j != 0:
            if map[i+1][j-1] == 0: # левая нижняя точка
              weights[width * i + j][width * (i + 1) + j - 1] = 1.41
            else:
              weights[width * i + j][width * (i + 1) + j - 1] = np.inf

          if i != 0 and j != 0:
            if map[i-1][j-1] == 0: # левая верхняя точка
              weights[width * i + j][width * (i - 1) + j - 1] = 1.41
            else:
              weights[width * i + j][width * (i - 1) + j - 1] = np.inf

  return weights

def find_new_index_astar(map, last_line, end, evr, heuristic_evaluation):
  # Найти следующую вершину
  line = np.copy(last_line)
  line[line == 0] = np.inf
  line_with_evr = line + heuristic_evaluation
  line_0 = np.sort(line_with_evr)
  for x in line_0:
    index = np.where(line_with_evr == x)[0][0]
    return index

def A_star(map, table, start, end, evr):
  # Заполнение таблицы
  heuristic_evaluation = evristick_line(map, table[0], end, evr);
  line = np.ones(len(table)) * np.inf
  index = start
  line[start] = 0

  table_astar = np.zeros([len(table)-1, len(table)])

  for i in range(len(line)-1):
    new_line = iteration_of_astar(index, line, table)
    table_astar[i] = new_line
    if index == end[0] * len(map[0]) + end[1]:
      break
    index = find_new_index_astar(map, new_line, end, evr, heuristic_evaluation)
    line = new_line
  return table_astar

def find_new_ceng_from_zero_astar(line, last_line):
  # Найти индекс для текущего шага
  index_0 = np.where(line == 0)
  for x in index_0[0]:
    if last_line[x] - line[x]:
      return x


def find_way_astar(table, start, end):
  # Восстановление пути
  index = end
  for i in range(len(table)):
    j = len(table) - i - 1
    if table[j][index] != 0:
      break

  way = [end]
  length = table[j][index]

  for i in range(j+1):
    k = j - i
    if table[k][index] != length:
      index = find_new_ceng_from_zero_astar(table[k+1], table[k])
      length = table[k][index]
      way.append(index)
  return way


def A_star_final(map, start_coord, end_coord, evr):
  table_1 = table_of_map_astar(map, evr)
  start_x, start_y = start_coord
  end_x, end_y = end_coord
  start = start_x + start_y*len(map[0])
  end = end_x + end_y*len(map[0])
  table_2 = A_star(map, table_1, start, end_coord, evr)
  way = find_way_astar(table_2, start, end)

  new_map = np.copy(map)

  x_y = []
  for z in way:
    x = z%len(map[0])
    y = z//len(map[0])
    new_map[y, x] = 2
    x_y.append((x, y))

  plt.imshow(new_map, cmap='gray')
  plt.show()
  return x_y
#######################################################################
def get_line(start, end, final_map):
    # Функция, реализующая алгоритм Брезенхема для рисования наклонных отрезков

    # Начальные условия
    x_s, y_s = start
    x_e, y_e = end
    dx = x_e - x_s
    dy = y_e - y_s
    sw = False
    is_steep = False

    # Поворот линии, если наклон больше 45гр
    if math.fabs(dy) > math.fabs(dx):
      is_steep = True
      x_s, y_s = y_s, x_s
      x_e, y_e = y_e, x_e
      dx = x_e - x_s
      dy = y_e - y_s

    # Перемена местами начальной и конечной точки, если начальная правее конечной
    if x_s > x_e:
      x_s, x_e = x_e, x_s
      y_s, y_e = y_e, y_s
      dx = x_e - x_s
      dy = y_e - y_s
      sw = True

    # Вычисление ошибки
    error = int(dx / 2.0)

    if y_s < y_e:
      ystep = 1
    else:
      ystep = -1

    # Создание точек между началом и концом
    y = y_s
    points = []
    for x in range(x_s, x_e + 1):
      if x >= 0 and y >= 0 and x <= len(final_map[0])  and y <= len(final_map[0]):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
      error -= math.fabs(dy)
      if error < 0:
        y += ystep
        error += dx

    # Переворот массива, если начальная и конечная точки были поменяны местами
    if sw:
        points.reverse()

    return points

def is_obstacle(point, obstacle_map):
  # Проверка на препятствие
  y, x = point
  if obstacle_map[y][x] == 1.0:
    return True
  else:
    return False


def in_goal_area(node, goal, goal_area):
  # Проверка на попадание в целевую зону
  distance_squared = (node[0] - goal[0])**2 + (node[1] - goal[1])**2
  return distance_squared <= goal_area**2

def random_point(map):
  # Генерация рандомной точки
  x = np.random.randint(1, len(map[0]))
  y = np.random.randint(1, len(map))
  return (y, x)

def find_near_node(new_node, nodes):
  # Поиск ближайшей ноды
  min_distance = np.inf
  near_node_coords = [0, 0]
  # Поиск ближайшей точки на дереве
  index = 0
  for node in nodes:
    if min_distance > math.sqrt((node[0] - new_node[0])**2 + (node[1] - new_node[1])**2):
      min_distance = math.sqrt((node[0] - new_node[0])**2 + (node[1] - new_node[1])**2)
      near_node_coords = (node[0], node[1])
      index = nodes.index(node)
  return [near_node_coords, index]

def rad_dynamic(initial_rad, len_Nodes, coefficient):
  # Обновление радиуса для RRT*
  return initial_rad * coefficient ** (len_Nodes)

def find_index(node, all_nodes):
  # Индекс ноды
  for i in range(len(all_nodes)):
    if all_nodes[i] == node:
      return i
      break

def parent_points(new_node, radius, ind_nearest, all_indexes, all_nodes):
  # Поиск родительских нод внутри радиуса
  mas_parent_points = [ind_nearest]
  # if len(all_indexes)  != 0:
  while True:
    for i in range(len(all_indexes)):
      if all_indexes[i][1] == mas_parent_points[-1]:
        if in_goal_area(new_node, all_nodes[all_indexes[i][0]], radius):
          mas_parent_points.append(all_indexes[i][0])
          break
        else:
          return mas_parent_points
      elif i == len(all_indexes) - 1:
        return mas_parent_points

def RRT_star(inp_map, start, goal, goal_area, delta):
  map = np.copy(inp_map)
  tree = []
  nodes = [start]
  indexes = []
  num_child_max = 0
  trajec = []

  while True:
    new_node = random_point(map)
    # Проверка сгенерированной точки на препятствие
    if is_obstacle(new_node, map):
      continue
    near_node, ind_par = find_near_node(new_node, nodes)
    # # Построение линии от ближайшей точки на дереве до сгенерированной
    line = get_line(near_node, new_node, inp_map) # (y, x)

    # Фактор роста
    if len(line) > delta:
      # Если длина линии больше, чем фактор роста
      new_node = line[delta]
      line = line[:delta+1]

    # Проверка на пересечение линией препятствия
    if any(is_obstacle(point, map) for point in line):
      continue

    # Поиск всех родителей ближайшей точки в окрестности
    if len(tree) != 0:
      par_points = parent_points(new_node, rad_dynamic(1000, len(nodes), 0.99), find_index(near_node, nodes), indexes, nodes)
      # Поиск наиболее раннего родителя из найденных, до которого можно провести отрезок
      for i in reversed(range(len(par_points))):
        line = get_line(nodes[par_points[i]], new_node, inp_map)

        if not any(is_obstacle(point, map) for point in line):
          near_node = nodes[par_points[i]]
          ind_par = par_points[i]
          break

    nodes.append(new_node)
    ind_child = len(nodes) - 1
    tree.append(line)
    indexes.append((ind_par, ind_child))

    if in_goal_area(new_node, goal, goal_area):
      line = get_line(new_node, goal, inp_map) # (y, x)
      # Проверка на пересечение линией препятствия
      if not(any(is_obstacle(point, map) for point in line)):
        tree.append(line)
        last_point = new_node
        break
  par_points_of_last = parent_points(last_point, 800, find_index(last_point, nodes), indexes, nodes)
  trajec.append(goal)
  for i in range(len(par_points_of_last)):
    trajec.append(nodes[par_points_of_last[i]])
  return tree, last_point, trajec


def RRT_star_main(map, start, goal, goal_area, delta):
  RRT_map = np.copy(map)
  if RRT_map[start[0]][start[1]] == 1 or RRT_map[goal[0]][goal[1]] == 1:
    print("ERROR")
  else:
    tree, last_point, trajec = RRT_star(RRT_map, start, goal, goal_area, delta)

  for i in range(len(tree)):
    for j in range(len(tree[i])):
      RRT_map[tree[i][j][0]][tree[i][j][1]] = 1

  plt.plot(start[1], start[0], 'bo')  # Стартовая позиция
  plt.plot(goal[1], goal[0], 'go')    # Целевая позиция

  # Отображаем карту с препятствиями
  plt.imshow(RRT_map, cmap='gray')
  X_values = [Point[1] for Point in trajec]
  Y_values = [Point[0] for Point in trajec]
  plt.plot(X_values, Y_values, "red", alpha = 0.5)
  plt.show()
  return trajec
#######################################################################

open("log.txt", "a")
os.remove("log.txt")
robot = KUKA('192.168.88.25', log=["log.txt", 1])
#robot = ['192.168.88.21', '192.168.88.22', '192.168.88.23', '192.168.88.24', '192.168.88.25']
# sim = GuiControl(1200, 900, robot)
# sim.run()

pygame.init()
screen = pygame.display.set_mode((100, 100))
v_speed = 0.1
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if v_speed < 1: v_speed += 0.1
            if event.key == pygame.K_DOWN:
                if v_speed > 0.2: v_speed -= 0.1
            if event.key == pygame.K_w:
                robot.move_base(f=v_speed, s=0, r=0)
            if event.key == pygame.K_d:
                robot.move_base(f=0, s=0, r=-v_speed)
            if event.key == pygame.K_a:
                robot.move_base(f=0, s=0, r=v_speed)
            if event.key == pygame.K_s:
                robot.move_base(f=-v_speed, s=0, r=0)
            if event.key == pygame.K_f:
                robot.move_base(f=0, s=v_speed, r=0)
            if event.key == pygame.K_c:
                robot.move_base(f=0, s=-v_speed, r=0)
            if event.key == pygame.K_m:
                _, _, _, _, map_check = paint_map(100, 1.3, 0.4, 25)
                plt.imshow(map_check, cmap='gray')
                plt.show()
            if event.key == pygame.K_i:
                x_min, y_min, x_rob, y_rob, map_for_algs = paint_map(100, 1.3, 0.4, 25)
                print("Выбран алгоритм RRT-star")
                plt.imshow(map_for_algs, cmap='gray')
                plt.plot(x_rob * 100 - x_min, y_rob * 100 - y_min, 'orange', marker=".")
                plt.show()
                x_fin, y_fin = int(input("Введите координату x: ")), int(input("Введите координату y: "))
                way = RRT_star_main(np.copy(map_for_algs), (int(y_rob * 100 - y_min), int(x_rob * 100 - x_min)), (y_fin, x_fin), 100, 100)
                way.reverse()
                x_last = x_rob
                y_last = y_rob
                for i in range(len(way)):
                    x_goto = calc_coords100(way[i][1], x_min)
                    y_goto = calc_coords100(way[i][0], y_min)
                    angle = math.atan2(y_goto-y_last, x_goto-x_last)
                    robot.go_to(x_goto, y_goto, angle)
                    while round(robot.increment_data[0], 2) != round(x_goto, 2) and round(robot.increment_data[1], 2) != round(y_goto, 2) and round(robot.increment_data[2], 2) != round(angle, 2):
                        time.sleep(0.1)
                    x_last = x_goto
                    y_last = y_goto
            if event.key == pygame.K_u:
                x_min, y_min, x_rob, y_rob, map_for_algs = paint_map(10, 1.05, 0.75, 3)
                print("Выбран алгоритм А-star")
                plt.imshow(map_for_algs, cmap='gray')
                plt.plot(x_rob * 10 - x_min, y_rob * 10 - y_min, 'orange', marker=".")
                plt.xticks(np.arange(0, len(map_for_algs[0]), 5))
                plt.yticks(np.arange(0, len(map_for_algs), 5))
                plt.grid()
                plt.show()
                x_fin, y_fin, method = int(input("Введите координату x: ")), int(input("Введите координату y: ")), int(input("0-Манхэттен, 1-Чебышев, 2-Евклид: "))
                way = A_star_final(np.copy(map_for_algs), [int(x_rob * 10 - x_min), int(y_rob * 10 - y_min)], [x_fin, y_fin], method)
                way.reverse()
                x_last = x_rob
                y_last = y_rob
                for i in range(len(way)):
                    x_goto = calc_coords10(way[i][0], x_min)
                    y_goto = calc_coords10(way[i][1], y_min)
                    angle = math.atan2(y_goto-y_last, x_goto-x_last)
                    robot.go_to(x_goto, y_goto, angle)
                    time.sleep(1)
                    x_last = x_goto
                    y_last = y_goto
        if event.type == pygame.KEYUP:
            robot.move_base(f=0, s=0, r=0)
    pygame.display.flip()