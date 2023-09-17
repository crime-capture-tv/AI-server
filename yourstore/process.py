
#
# person_data = {
#     '20230912_13:50:00':'normal',
#     '20230912_13:50:04':'normal',
#     '20230912_13:50:08':'normal',
#     '20230912_13:50:12':'catch',
#     '20230912_13:50:16':'normal',
#     '20230912_13:50:20':'put',
#     '20230912_13:50:24':'normal',
#     '20230912_13:50:28':'catch',
#     '20230912_13:50:32':'normal',
#     '20230912_13:50:36':'catch',
#     '20230912_13:50:40':'catch',
#     '20230912_13:50:44':'catch',
#     '20230912_13:50:48':'normal'
# }
#
# counter_time = {
#     'startTime':'20230912_14:44:54',
#     'endTime':'20230912_14:45:10'
# }

class Process():
    def check_behavior(person_data, counter_time):
        # 계산대에서 머무른 시간 계산
        from datetime import datetime
        format_str = "%Y%m%d_%H:%M:%S"
        start_time = datetime.strptime(counter_time['startTime'], format_str)
        end_time = datetime.strptime(counter_time['endTime'], format_str)
        duration = (end_time - start_time).seconds

        # 행동 데이터 분석
        previous_action = None
        first_catch_time = start_time  # 첫 'catch'의 시간을 저장하기 위한 변수

        for time, action in person_data.items():
            if previous_action == 'catch' and action == 'normal':
                if duration >= 8:
                    print('1')
                    return 1, None
                else:
                    print('2')
                    return 2, first_catch_time
            elif previous_action == 'catch' and action == 'put':
                print('1')
                return 1, None
            elif action == 'insert':
                if duration >= 8:
                    print('1')
                    return 1, None
                else:
                    print('2')
                    return 2, first_catch_time
            previous_action = action

        # catch 없이 normal만 있을 경우
        if 'catch' not in person_data.values():
            print('1')
            return 1, None
        print('2')
        return 2, first_catch_time
