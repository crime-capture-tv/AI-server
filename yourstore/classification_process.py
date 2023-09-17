from datetime import datetime


class ClassificationProcess():
    def __init__(self) -> None:
        pass
        

    def check_behavior(self, person_data, counter_time, format_str):
        start_time = datetime.strptime(counter_time['startTime'], format_str)
        end_time = datetime.strptime(counter_time['endTime'], format_str)
        duration = (end_time - start_time).seconds

        calculation_check = False
        catch_times = []

        for time, action in person_data.items():
            # print(f'{time} - {action}')
            if action[1] == 'catch':
                calculation_check = True
                catch_times.append([time, action[0]])
            elif action[1] == 'put':
                calculation_check = False
            elif action[1] == 'insert':
                calculation_check = True
                catch_times.append([time, action[0]])
            elif action[1] == 'normal':
                pass
            
            action_time = datetime.strptime(time, format_str)
            if start_time < action_time < end_time and calculation_check:
                if duration >= 6:
                    calculation_check = False

        if len(catch_times)==0:
            catch_times = [[None, None], [None, None]] 
        elif len(catch_times)==1:
            catch_times = [catch_times[0], catch_times[0]] 
        else:
            catch_times = [catch_times[0], catch_times[-1]] 

        # if calculation_check:
        #     return 'Warning', catch_times
        # else:
        #     return 'Clear', catch_times

        return 'Warning', catch_times
        # return 'Clear', catch_times


if __name__ == '__main__':
    person_data = {
        '20230912_13:50:00':'normal',
        '20230912_13:50:04':'normal',
        '20230912_13:50:08':'normal',
        '20230912_13:50:12':'catch',
        '20230912_13:50:16':'normal',
        '20230912_13:50:20':'put',
        '20230912_13:50:24':'normal',
        '20230912_13:50:28':'catch',
        '20230912_13:50:32':'normal',
        '20230912_13:50:36':'catch',
        '20230912_13:50:40':'catch',
        '20230912_13:50:44':'catch',
        '20230912_13:50:48':'normal',
        '20230912_13:50:52':'normal',
        '20230912_13:50:56':'normal',
        '20230912_13:51:00':'normal',
        '20230912_13:51:04':'catch',
        '20230912_13:51:08':'normal',
    }

    counter_time = {
        'startTime':'20230912_13:50:37',
        'endTime':'20230912_13:50:45'
    }

    classification_process = ClassificationProcess()
    result = classification_process.check_behavior(person_data, counter_time)
    print(result)