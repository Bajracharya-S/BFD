# example from https://www.codecademy.com/resources/docs/python/threading

import threading, time, random

# Simulates waiting time (e.g., an API call/response)
def slow_function(thread_index):
  time.sleep(random.randint(1, 10))
  print("Thread {} done!".format(thread_index))

def run_threads():
  threads = []

  for thread_index in range(5):
    individual_thread = threading.Thread(target=slow_function, args=(thread_index,))
    threads.append(individual_thread)
    individual_thread.start()

  # At this point, threads are running independently from the main flow of application and each other
  print("Main flow of application")

  # This ensures that all threads finish before the main flow of application continues
  for individual_thread in threads:
    individual_thread.join()

  print("All threads are done")

run_threads()
