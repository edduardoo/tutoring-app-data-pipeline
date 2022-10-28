import numpy as np
import pandas as pd
from unittest import main, TestCase

class StudentsGenerator():
    def __init__(self, size):
        self._size = size
        self._students = pd.DataFrame({"student_id": np.arange(1, size + 1)})
        self._students['weight'] = np.random.exponential(1, size=size) # the avg weight by student is 1 but most of them weight less (exponential distribution)

    def get_size(self):
        return self._size

    def get_students(self):
        return self._students[['student_id']] # returning only public columns        
        
    

class TestStudents(TestCase):
    def test_if_creates_df_correctly(self):
        size = 10
        gen = StudentsGenerator(size)
        result_df, expected_df = gen.get_students(), pd.DataFrame
        result_size, expected_size = gen.get_size(), size

        self.assertIsInstance(result_df, expected_df)
        self.assertEqual(result_size, expected_size)

if __name__ == '__main__':
    main()