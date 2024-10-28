# main.py

import logging
from core.admin_kitty import КотикАдмин

logging.basicConfig(level=logging.INFO)

def main():
    котик_админ = КотикАдмин()
    print("Мяу-мяу! КотикАдмин инициализирован с моделью Mistral 7B. Напишите 'выход', чтобы завершить работу.")
    
    while True:
        команда = input("Человек: ")
        if команда.lower() == 'выход':
            break
        ответ = котик_админ.обработать_команду(команда)
        print(f"КотикАдмин: {ответ}")

if __name__ == "__main__":
    main()
