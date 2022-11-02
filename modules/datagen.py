import numpy as np
import pandas as pd
from unittest import main, TestCase
from datetime import datetime, timedelta
from modules.helpers import ECDF
from faker import Faker

class StudentsGenerator():
    def __init__(self, size, created_at = None):
        if created_at is None:
            created_at = datetime.now()
        faker = Faker()
        self._size = size
        self._students = pd.DataFrame({"student_id": np.arange(100001, size + 100001)})
        names = [faker.name() for i in range(size)]        
        self._students['name'] = np.array(names)
        self._students['weight'] = np.random.exponential(1, size=size) # the avg weight by student is 1 but most of them weight less (exponential distribution)
        self._students['created_at'] = created_at
        self._students['updated_at'] = created_at
        

    def get_size(self):
        return self._size

    def get_students(self):
        return self._students.drop(['weight'], axis=1) # returning only public columns        
        

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
        self._next_session_id = 100000
        
    def get_sessions(self):
        return self._sessions.drop(['duration_min', 'wait_time'], axis=1)


    def _generate_sessions_for_date(self, date):
                
        number_of_sessions = self._scheduler.generate_sessions_count(date, self._students_gen.get_size())
        if number_of_sessions == 0:
            return None
        students_sample = self._students_gen._students.sample(n=number_of_sessions, replace=True, weights='weight', ignore_index=True)
        sessions = pd.DataFrame({'session_id': range(self._next_session_id, self._next_session_id + number_of_sessions), 'date': np.full(number_of_sessions, date)})
        self._next_session_id = self._next_session_id + number_of_sessions
        sessions = pd.concat([sessions, students_sample], axis=1)
        sessions['hours'] = self._times_generator.sample_session_times(number_of_sessions)       
        sessions['started_at'] = sessions.apply(lambda row: row['date'] + timedelta(hours=row['hours']), axis=1)                
        sessions['subject_id'] = self._subjects_gen.sample(len(sessions))
        sessions['duration_min'] = np.random.gamma(4, 4, size=len(sessions)) # session duration, inline sampling
        sessions['status'] = 'Queued'
        sessions['tutor_id'] = None
        sessions['connected_at'] = None
        sessions['connected_at'] = sessions['connected_at'].astype('datetime64[ns]')
        sessions['wait_time'] = None
        sessions['wait_time'] = sessions['wait_time'].astype('timedelta64[ns]')
        sessions['finished_at'] = None
        sessions['finished_at'] = sessions['finished_at'].astype('datetime64[ns]')

        return sessions[['session_id', 'started_at', 'student_id', 'subject_id', 'duration_min', 'status', 'tutor_id', 'connected_at', 'finished_at']]

    def generate_sessions_for_date_range(self, start_date, end_date):
        date_range = pd.date_range(start_date, end_date)
        sessions_range = pd.DataFrame()
        for date in date_range:            
            sessions_date_tmp = self._generate_sessions_for_date(date)            
            sessions_range = pd.concat([sessions_range, sessions_date_tmp], axis=0).reset_index(drop=True)
            
        self._sessions = sessions_range

    def process(self, frequence):
        if self._sessions is None:
            print("Sessions have not been generated yet!")
            return        

        # PERIODIC PROCESS:
        start_at = self._sessions['started_at'].min()
        end_at = self._sessions['started_at'].max() + timedelta(hours=2) # keep processing after 2 hours to make sure everyone is processed                
        date_range = pd.date_range(start_at, end_at, freq=frequence) # (e.g. '10S' means periods of 10 seconds)
        date_range = date_range.to_pydatetime()
        shifts_changes = self._scheduler.work_shifts_for_date_range(start_at, end_at)
        shifts_changes = iter(shifts_changes)
        next_shift_change = next(shifts_changes)
                
        counter = 0
        total = len(date_range)
        for time in date_range:
            counter += 1
            print(str(round(counter/total*100)) + '%', end='\r')

            # check_finished sessions
            self.check_finished_sessions(time)

            # check tutors to unblock
            #self._tutors_gen.check_tutors_to_unblock()

            # check_tutors_status
            if next_shift_change is not None and time >= next_shift_change:
                #print("Shift change: ", next_shift_change, " AT ", time) 
                self._tutors_gen.check_tutors_statuses(time)
                try:
                    next_shift_change = next(shifts_changes)
                except:
                    #print("Shifits finished!")
                    next_shift_change = None

            
            # this query makes sure we revisit sessions that could not have a tutor assigned
            sessions_to_process = self._sessions.query("@time >= started_at and status == 'Queued'")
            
            # PROCESSING QUEUED SESSIONS:
            # ----------------------------            
            for idx, row in sessions_to_process.iterrows():
                #print("Processing session:", idx)            
                # assign a tutor
                tutor_id = self._tutors_gen.assign_a_tutor(row['subject_id'])                
                if tutor_id is not None:
                    #print("Processing session:", idx) 
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
        samples = sorted(samples)
        return samples


class SubjectsGenerator():
    def __init__(self, created_at = None):
        if created_at is None:
            created_at = datetime.now()
        # demand is a number between 1-5 which represents the level of demand for a subject
        self._subjects = pd.DataFrame({
                                  'subject_id': range(100, 104),
                                  'subject': ['Math', 'English', 'Biology', 'Chemistry'],
                                  'demand':  [5, 4, 2, 3],
                                  'created_at': np.full(4, created_at),
                                  'updated_at': np.full(4, created_at)
                                })

    def get_subjects(self):
        return self._subjects
    
    def sample(self, size):
        samples = self._subjects.sample(size, replace=True, weights='demand')['subject_id'].reset_index(drop=True)
        return samples

class TutorsGenerator():
        
    MAX_SESSIONS_BY_TUTOR = 5
    
    def __init__(self, size, subjects_gen, created_at = None):
        if created_at is None:
            created_at = datetime.now()
        faker = Faker()  
        names = [faker.name() for i in range(size)]                
        self._subjects_gen = subjects_gen
        self._tutors = pd.DataFrame({
                        'tutor_id': np.arange(10001, size + 10001),
                        'name': np.array(names),
                        'status': np.full(size, 'Offline'),
                        'active_sessions': np.full(size, 0)
                       })
        self._tutors['subject_id'] = subjects_gen.sample(size)
        self._tutors = self._tutors.merge(subjects_gen.get_subjects()[['subject_id', 'subject']])
        self._scheduler = Scheduler()        
        self._tutors['work_shift_id'] = self._scheduler.schedule_work_shifts(size)
        self._tutors['rest_day'] = self._scheduler.schedule_rest_day(size)
        self._tutors = self._tutors.merge(self._scheduler._work_shifts[['work_shift_id', 'starts_at', 'ends_at']]) # tutors full
        self._tutors['created_at'] = created_at
        self._tutors['updated_at'] = created_at
        self._tutors.set_index('tutor_id', inplace=True)
        
    def get_tutors_full(self):
        return self._tutors

    def get_tutors(self):
        return self._tutors.drop(['subject', 'starts_at', 'ends_at'], axis=1).reset_index()
    
    def get_work_shifts(self):
        return self._scheduler._work_shifts
    
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
    
    def assign_a_tutor(self, subject_id):
        min_sessions_active = self._tutors.query("status == 'Available' and subject_id == @subject_id")['active_sessions'].min()
        tutors_min = self._tutors.query("status == 'Available' and subject_id == @subject_id and active_sessions == @min_sessions_active")
        
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
        
        # Signing_off
        if self._tutors.loc[tutor_id, 'status'] == 'Leaving' and self._tutors.loc[tutor_id, 'active_sessions'] == 0:
            self._tutors.loc[tutor_id, 'status'] = 'Offline'



class Scheduler():
    def __init__(self, created_at = None):
        if created_at is None:
            created_at = datetime.now()
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
                            'work_shift_id': range(1000, 1003),
                            'label': ['00am-08am', '08am-04pm', '04pm-00am'],
                            'percent_tutors':  [0.05, 0.45, 0.5], 
                            'starts_at': ['00:00:00', '08:00:00', '16:00:00'],
                            'ends_at':   ['07:59:59', '15:59:59', '23:59:59'],
                            'created_at': np.full(3, created_at),
                            'updated_at': np.full(3, created_at)
                            })

        # rest days: generating the probabilities for a tutor to rest in each day of the week, based on _avg_pct_sessions_by_day
        days, weights = zip(*self._avg_pct_sessions_by_day.items())
        weights = [1/w for w in weights] # inverting the weight of the probability
        self._rest_days = pd.DataFrame({'day': days, 'weight': weights})             

    def work_shifts_for_date_range(self, start_dt, end_dt):
        distinct_shifts = self._work_shifts['starts_at'].unique()
        times = [datetime.strptime(time_string, '%H:%M:%S').time() for time_string in distinct_shifts]
        deltas = [timedelta(hours=time.hour, minutes=time.minute, seconds=time.second) for time in times]
        date_range = pd.date_range(start_dt.date(), end_dt.date())
        shifts = [dt + delta for dt in date_range for delta in deltas]
        return shifts

    def generate_sessions_count(self, date, students_count):
        day_of_week = date.strftime('%A')
        avg_pct_sessions = self._avg_pct_sessions_by_day[day_of_week]
        avg_sessions = int(avg_pct_sessions * students_count)
        return np.random.poisson(avg_sessions)
    
    # sets the work shift and the rest day for each tutor
    def schedule_work_shifts(self, size):
        work_shifts = self._work_shifts.sample(size, replace=True, weights='percent_tutors').reset_index(drop=True)
        #work_shifts = work_shifts[['label', 'starts_at', 'ends_at']].reset_index(drop=True)
        #return pd.concat([tutors, work_shifts], axis=1)
        return work_shifts['work_shift_id']

    def schedule_rest_day(self, size):
        return self._rest_days.sample(size, replace=True, weights='weight')['day'].reset_index(drop=True)
   
        

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
