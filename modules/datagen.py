import numpy as np
import pandas as pd
from unittest import main, TestCase
from datetime import datetime, timedelta
from modules.helpers import ECDF

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

class SessionsGenerator():

    def __init__(self, students_gen, subjects_gen, tutors_gen):
        self._students_gen = students_gen
        self._subjects_gen = subjects_gen
        self._tutors_gen = tutors_gen
        self._scheduler = Scheduler()
        self._times_generator = SessionTimesGenerator()
        self._sessions = None
        
    def get_sessions(self):
        return self._sessions

    def _generate_sessions_for_date(self, date):
                
        number_of_sessions = self._scheduler.generate_sessions_count(date, self._students_gen.get_size())
        if number_of_sessions == 0:
            return None
        students_sample = self._students_gen._students.sample(n=number_of_sessions, replace=True, weights='weight', ignore_index=True)
        sessions = pd.DataFrame({'date': np.full(number_of_sessions, date)})
        sessions = pd.concat([sessions, students_sample], axis=1)
        sessions['hours'] = self._times_generator.sample_session_times(number_of_sessions)       
        sessions['started_at'] = sessions.apply(lambda row: row['date'] + timedelta(hours=row['hours']), axis=1)                
        sessions['subject'] = self._subjects_gen.sample(len(sessions))
        sessions['duration_min'] = np.random.gamma(4, 4, size=len(sessions)) # session duration, inline sampling
        sessions['status'] = 'Queued'
        sessions['tutor_id'] = None
        sessions['connected_at'] = None
        sessions['connected_at'] = sessions['connected_at'].astype('datetime64[ns]')
        sessions['wait_time'] = None
        sessions['wait_time'] = sessions['wait_time'].astype('timedelta64[ns]')
        sessions['finished_at'] = None
        sessions = sessions.sort_values('started_at')

        return sessions[['started_at', 'student_id', 'subject', 'duration_min', 'status', 'tutor_id', 'connected_at', 'finished_at']]

    def generate_sessions_for_date_range(self, start_date, end_date):
        date_range = pd.date_range(start_date, end_date)
        sessions_range = pd.DataFrame()
        for date in date_range:            
            sessions_date_tmp = self._generate_sessions_for_date(date)            
            sessions_range = pd.concat([sessions_range, sessions_date_tmp], axis=0).reset_index(drop=True)
            
        self._sessions = sessions_range

    def process(self):
        if self._sessions is None:
            print("Sessions have not been generated yet!")
            return

        # TODO: PROBLEMA PRA RESOLVER: se um tutor desocupar e soh percebermos 2 horas depois pq teve um gap de sessions, 
        # qual horario vamos colocar no "connected_at", se eu nao salvo o horario que um tutor baixou de 5 atendimentos?                
        # nao é melhor fazer um loop nos segundos e verificar de segundo em segundo? ou a cada 5 segundos? 
        for time in self._sessions['started_at']:
            
            # check_finished sessions
            self.check_finished_sessions(time)

            # check tutors to unblock
            #self._tutors_gen.check_tutors_to_unblock()

            # check_tutors_status
            self._tutors_gen.check_tutors_statuses(time)
            
            # this query makes sure we revisit sessions that could not have a tutor assigned
            sessions_to_process = self._sessions.query("@time >= started_at and status == 'Queued'")
            
            # PROCESSING QUEUED SESSIONS:
            # ----------------------------            
            for idx, row in sessions_to_process.iterrows():
                #print("Processing session:", idx)            
                # assign a tutor
                tutor_id = self._tutors_gen.assign_a_tutor(row['subject'])                
                if tutor_id is not None:
                    self._sessions.loc[idx, 'tutor_id'] = tutor_id
                    self._sessions.loc[idx, 'status'] = 'Active'
                    self._sessions.loc[idx, 'connected_at'] = time
                    self._sessions.loc[idx, 'finished_at'] = time + timedelta(minutes=row['duration_min'])
        
        # finish processing
        self._sessions['wait_time'] = self._sessions['connected_at'] - self._sessions['started_at']

    

    def check_finished_sessions(self, datetime_now):
        sessions_to_finish = self._sessions.query("status == 'Active' and finished_at < @datetime_now")
        #print("Sessions to finish:", sessions_to_finish.index)
        self._sessions.loc[sessions_to_finish.index, 'status'] = 'Finished'        
        for idx, row in sessions_to_finish.iterrows():
            self._tutors_gen.unassign_a_tutor(row['tutor_id'])
        
    

class TestSessions(TestCase):
    def test_if_creates_df_correctly(self):
        size = 10
        gen = StudentsGenerator(size)
        result_df, expected_df = gen.get_students(), pd.DataFrame
        result_size, expected_size = gen.get_size(), size

        self.assertIsInstance(result_df, expected_df)
        self.assertEqual(result_size, expected_size)

# TODO: move to Scheduler
class SessionTimesGenerator():
    
    def __init__(self, 
                 peaks_hours=[10, 15, 20],  # students who go to school in the evening (tutoring in the morning), in the morning (tutoring in the evening) and all students (tutoring at night)
                 peaks_deviations=[2, 2, 2],
                 peaks_weights=[1, 1, 1.5],
                 random_state=877
                 ):
        self.random_state = random_state
        self._multi_modal_dist = np.array([])
        for i in range(len(peaks_hours)):
            np.random.seed(self.random_state)
            peak_temp = np.random.normal(peaks_hours[i], peaks_deviations[i], size=int(peaks_weights[i]*100000))
            self._multi_modal_dist = np.concatenate([self._multi_modal_dist, peak_temp])
        
        self._multi_modal_dist = np.array([x-24 if x>=24 else 24+x if x<0 else x for x in self._multi_modal_dist])
        self._ecdf = ECDF(self._multi_modal_dist)        
                


    def sample_session_times(self, size):
        # sampling from an empirical distribution:
        # we first draw a sample from the uniform cont. distribution [0,1]. Then we use this random number between 0 and 1
        # as a percentile to get the corresponding value from our ecdf. 
        # This operation is vectorized here: 
        random_percentiles = np.random.random(size)
        samples = [self._ecdf.ppf(r) for r in random_percentiles]
        return samples


class SubjectsGenerator():
    def __init__(self):
        # demand is a number between 1-5 which represents the level of demand for a subject
        self._subjects = pd.DataFrame({
                                  'subject': ['Math', 'English', 'Biology', 'Chemistry'],
                                  'demand':  [5, 4, 2, 3]
                                })

    def sample(self, size):
        samples = self._subjects.sample(size, replace=True, weights='demand')['subject'].reset_index(drop=True)
        return samples

class TutorsGenerator():
        
    MAX_SESSIONS_BY_TUTOR = 5
    
    def __init__(self, size, subjects_gen):
        self._subjects_gen = subjects_gen
        self._tutors = pd.DataFrame({
                        'tutor_id': np.arange(1, size + 1),
                        'status': np.full(size, 'Offline'),
                        'active_sessions': np.full(size, 0)
                       })
        self._tutors['subject'] = subjects_gen.sample(size)
        self._scheduler = Scheduler()
        self._tutors = self._scheduler.schedule_work_shifts(self._tutors)
        self._tutors.set_index('tutor_id', inplace=True)
        
    def get_tutors(self):
        return self._tutors
    
    def sample(self):
        pass

    def check_tutors_statuses(self, datetime_now):
        day_of_week = datetime_now.strftime('%A')
        currentTime = datetime_now.strftime("%H:%M:%S")
        
        tutors_leaving = self._tutors.query("status == 'Available' and @currentTime >= ends_at")
        #print("Tutors Leaving", tutors_leaving.index)
        self._tutors.loc[tutors_leaving.index, 'status'] = 'Leaving'
        
        tutors_to_sign_off = self._tutors.query("status != 'Offline' and @currentTime >= ends_at and active_sessions == 0")
        #print("Tutors to sing off", tutors_to_sign_off.index)
        self._tutors.loc[tutors_to_sign_off.index, 'status'] = 'Offline'
        
        tutors_to_sign_in = self._tutors.query("status != 'Available' and @currentTime >= starts_at and @currentTime <= ends_at and @day_of_week != rest_day")
        #print("Tutors to sign in", tutors_to_sign_in.index)
        self._tutors.loc[tutors_to_sign_in.index, 'status'] = 'Available'
    
    def assign_a_tutor(self, subject):
        min_sessions_active = self._tutors.query("status == 'Available' and subject == @subject")['active_sessions'].min()
        tutors_min = self._tutors.query("status == 'Available' and subject == @subject and active_sessions == @min_sessions_active")
        
        if len(tutors_min) == 0: # no tutor available
            return None
                
        tutor = tutors_min.sample()
        self._tutors.loc[tutor.index[0], 'active_sessions'] = min_sessions_active + 1
        
        # Blocking
        if tutor['active_sessions'].iloc[0] == TutorsGenerator.MAX_SESSIONS_BY_TUTOR - 1:
            self._tutors.loc[tutor.index[0], 'status'] = 'Blocked'

        return tutor.index[0]
    
    def unassign_a_tutor(self, tutor_id):
        self._tutors.loc[tutor_id, 'active_sessions'] -= 1
        
        # Unblocking
        if self._tutors.loc[tutor_id, 'status'] == 'Blocked':
            self._tutors.loc[tutor_id, 'status'] = 'Available'            




class Scheduler():
    def __init__(self):
        self._avg_pct_sessions_by_day = { # avg sessions by day as a percentage of students
                        "Monday": 0.12,
                        "Tuesday": 0.13,
                        "Wednesday": 0.125,
                        "Thursday": 0.11,
                        "Friday": 0.07,
                        "Saturday": 0.02,
                        "Sunday": 0.05
                        }
        
        # tutors' different work shifts, with the percentage of tutors in each one, and the start and end times
        self._work_shifts = pd.DataFrame({
                            'working_hours': ['00-08am', '08am-04pm', '04pm-00'],
                            'percent_tutors':  [0.05, 0.45, 0.5], 
                            'starts_at': ['00:00:00', '08:00:00', '16:00:00'],
                            'ends_at':   ['07:59:59', '15:59:59', '23:59:59']
                            })

        # rest days: generating the probabilities for a tutor to rest in each day of the week, based on _avg_pct_sessions_by_day
        days, weights = zip(*self._avg_pct_sessions_by_day.items())
        weights = [1/w for w in weights] # inverting the weight of the probability
        self._rest_days = pd.DataFrame({'day': days, 'weight': weights})         
        
    
    def generate_sessions_count(self, date, students_count):
        day_of_week = date.strftime('%A')
        avg_pct_sessions = self._avg_pct_sessions_by_day[day_of_week]
        avg_sessions = int(avg_pct_sessions * students_count)
        return np.random.poisson(avg_sessions)
    
    # sets the work shift and the rest day for each tutor
    def schedule_work_shifts(self, tutors):
        work_shifts = self._work_shifts.sample(len(tutors), replace=True, weights='percent_tutors')
        work_shifts = work_shifts[['working_hours', 'starts_at', 'ends_at']].reset_index(drop=True)
        work_shifts['rest_day'] = self._rest_days.sample(len(tutors), replace=True, weights='weight')['day'].reset_index(drop=True)                                
        


        return pd.concat([tutors, work_shifts], axis=1)
        


if __name__ == '__main__':
    #main()
    st_gen = StudentsGenerator(100)
    sub_gen = SubjectsGenerator()
    gen = SessionsGenerator(st_gen, sub_gen)
    #tut_gen = TutorsGenerator(1000, sub_gen)
    #result = tut_gen.get_tutors()#.groupby('working_hours')['subject'].value_counts(normalize=True)
    #result = gen._generate_sessions_for_date(datetime.strptime('2022-10-15', '%Y-%m-%d'))
    
    gen.generate_sessions_for_date_range('2022-10-10', '2022-10-11')
    gen.process()
    
    #print(result)
    #import matplotlib.pyplot as plt
    #gen = SessionTimesGenerator()
    #samples = gen.sample_session_times(10000)
    #plt.hist(samples, bins=100)
    #plt.show()
