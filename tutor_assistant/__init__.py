"""AI-помощник тьютора: разбор и анализ наблюдений за детьми с ОВЗ.

Публичные функции:

* extract — извлечение полей из записи (Часть 1);
* classify — тип записи и уверенность (Часть 2);
* check_zone — отнесение наблюдения к зонам развития (Часть 3).
"""

from tutor_assistant.classify import classify
from tutor_assistant.extract import extract
from tutor_assistant.zones import check_zone

__all__ = ["check_zone", "classify", "extract"]
