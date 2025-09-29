import pandas as pd
from ortools.sat.python import cp_model

def solve_class_assignment(csv_path="class_feature.csv"):
    """
    학생 데이터를 CSV 파일에서 읽어와 최적의 반 배정 문제를 해결합니다.
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='cp949')
        except Exception as e:
            print(f"오류: CSV 파일을 읽는 중 에러가 발생했습니다. 인코딩을 확인해주세요. ({e})")
            return
    except FileNotFoundError:
        print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다. 파일 이름과 경로를 확인해주세요.")
        return

    # --- 1. 기본 데이터 설정 ---
    num_students = len(df)
    num_classes = 6
    students = list(df['id'])
    classes = list(range(num_classes))
    
    total_students = len(df)
    base_size = total_students // num_classes
    remainder = total_students % num_classes
    
    class_sizes = {}
    for i in range(num_classes):
        class_sizes[i] = base_size
    for i in range(remainder):
        class_sizes[i] += 1
        
    print(f"총 학생 수: {total_students}명")
    print(f"클래스별 정원: {class_sizes}")

    # --- 2. CP-SAT 모델 생성 ---
    model = cp_model.CpModel()

    # --- 3. 변수 정의 ---
    assign = {}
    for i in students:
        for j in classes:
            assign[(i, j)] = model.NewBoolVar(f'assign_s{i}_c{j}')

    # --- 4. 제약 조건 추가 ---
    for i in students:
        model.AddExactlyOne([assign[(i, j)] for j in classes])

    for j in classes:
        model.Add(sum(assign[(i, j)] for i in students) == class_sizes[j])
        
    for _, row in df.iterrows():
        s1 = row['id']
        s2 = row['나쁜관계']
        if pd.notna(s2):
            s2 = int(s2)
            if s2 in students:
                for j in classes:
                    model.AddBoolOr([assign[(s1, j)].Not(), assign[(s2, j)].Not()])

    for _, row in df.iterrows():
        s1 = row['id']
        s2 = row['좋은관계']
        if row['비등교'] == 'yes' and pd.notna(s2):
            s2 = int(s2)
            if s2 in students:
                for j in classes:
                    model.AddImplication(assign[(s1, j)], assign[(s2, j)])

    leaders = df[df['Leadership'] == 'yes']['id'].tolist()
    if leaders:
        for j in classes:
            model.Add(sum(assign[(i, j)] for i in leaders) >= 1)

    last_year_mates = []
    for last_class in df['24년 학급'].unique():
        last_year_class_group = df[df['24년 학급'] == last_class]['id'].tolist()
        for i, s1 in enumerate(last_year_class_group):
            for s2 in last_year_class_group[i+1:]:
                last_year_mates.append((s1, s2))

    overlap_vars = []
    if last_year_mates:
        for s1, s2 in last_year_mates:
            for j in classes:
                overlap = model.NewBoolVar(f'overlap_{s1}_{s2}_c{j}')
                model.Add(assign[(s1, j)] + assign[(s2, j)] == 2).OnlyEnforceIf(overlap)
                model.Add(assign[(s1, j)] + assign[(s2, j)] < 2).OnlyEnforceIf(overlap.Not())
                overlap_vars.append(overlap)
    
    total_overlaps = model.NewIntVar(0, len(last_year_mates) if last_year_mates else 0, 'total_overlaps')
    if overlap_vars:
        model.Add(total_overlaps == sum(overlap_vars))
    else:
        model.Add(total_overlaps == 0)

    # --- 5. 균등 분배를 위한 최적화 목표 설정 ---

    # 모든 패널티(편차) 항목들을 담을 리스트
    all_penalty_terms = []

    # 속성 정의 (CSV 파일의 컬럼명에 맞게 수정)
    properties = {
        '피아노': df[df['Piano'] == 'yes']['id'].tolist(),
        '비등교': df[df['비등교'] == 'yes']['id'].tolist(),
        '남성': df[df['sex'] == 'boy']['id'].tolist(),
        '여성': df[df['sex'] == 'girl']['id'].tolist(),
        '운동': df[df['운동선호'] == 'yes']['id'].tolist()
    }

    # 제약조건 3, 5, 6, 7: 특정 그룹 학생들을 균등하게 분배
    for prop_name, student_group in properties.items():
        if student_group:
            counts_per_class = [sum(assign[(i, j)] for i in student_group) for j in classes]
            max_count = model.NewIntVar(0, len(student_group), f'max_{prop_name}')
            min_count = model.NewIntVar(0, len(student_group), f'min_{prop_name}')
            model.AddMaxEquality(max_count, counts_per_class)
            model.AddMinEquality(min_count, counts_per_class)
            # 각 편차에 가중치 10 부여
            all_penalty_terms.append((max_count - min_count) * 10)

    # 제약조건 4: 성적을 균등하게 분배 (반별 성적 총합의 편차 최소화)
    score_map = df.set_index('id')['score'].to_dict()
    max_possible_score_sum = df['score'].sum()
    
    score_sums_per_class = []
    for j in classes:
        class_score_sum = model.NewIntVar(0, max_possible_score_sum, f'score_sum_c{j}')
        model.Add(class_score_sum == sum(assign[(i, j)] * score_map[i] for i in students))
        score_sums_per_class.append(class_score_sum)
    
    max_score_sum = model.NewIntVar(0, max_possible_score_sum, 'max_score_sum')
    min_score_sum = model.NewIntVar(0, max_possible_score_sum, 'min_score_sum')
    model.AddMaxEquality(max_score_sum, score_sums_per_class)
    model.AddMinEquality(min_score_sum, score_sums_per_class)
    score_deviation = max_score_sum - min_score_sum
    # 성적 편차는 가중치 1 부여 (기준점 역할)
    all_penalty_terms.append(score_deviation)

    # 제약조건 9: 클럽 활동 멤버가 편향되지 않도록 분배
    clubs = df['클럽'].dropna().unique()
    for club in clubs:
        club_members = df[df['클럽'] == club]['id'].tolist()
        if len(club_members) > 1:
            counts_per_class = [sum(assign[(i, j)] for i in club_members) for j in classes]
            max_count = model.NewIntVar(0, len(club_members), f'max_{club}')
            min_count = model.NewIntVar(0, len(club_members), f'min_{club}')
            model.AddMaxEquality(max_count, counts_per_class)
            model.AddMinEquality(min_count, counts_per_class)
            # 클럽 편차에 가중치 10 부여
            all_penalty_terms.append((max_count - min_count) * 10)

    # 작년 동급생 겹침 패널티 추가 (기존 가중치 5 -> 50으로 상향 조정)
    all_penalty_terms.append(total_overlaps * 50)

    # 모든 패널티의 합을 최종 최적화 목표로 설정
    total_penalty = model.NewIntVar(0, 100000, 'total_penalty') # 충분히 큰 값으로 설정
    model.Add(total_penalty == sum(all_penalty_terms))
    model.Minimize(total_penalty)

    # --- 6. 솔버 실행 및 결과 출력 ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("\n 최적의 반 배정 결과를 찾았습니다!")
        df['배정된반'] = 0
        for i in students:
            for j in classes:
                if solver.Value(assign[(i, j)]) == 1:
                    df.loc[df['id'] == i, '배정된반'] = j + 1
                    break
        
        print_summary(df, classes)
        
        df.to_csv("result.csv", index=False, encoding='utf-8-sig')
        print("\n결과가 'result.csv' 파일로 저장되었습니다.")

    elif status == cp_model.INFEASIBLE:
        print("\n 모든 제약 조건을 만족하는 해를 찾을 수 없습니다. 조건을 완화해 보세요.")
    else:
        print("\n해를 찾지 못했습니다. 탐색 시간이 더 필요할 수 있습니다.")


def print_summary(df, classes):
    """결과를 분석하고 요약하여 출력합니다."""
    print("\n--- 클래스별 배정 결과 요약 ---")
    for j in classes:
        class_num = j + 1
        class_df = df[df['배정된반'] == class_num]
        print(f"\n[ {class_num}반 ] (총 {len(class_df)}명)")
        print(f"  - 성별 (남/여): {len(class_df[class_df['sex']=='boy'])} / {len(class_df[class_df['sex']=='girl'])}")
        print(f"  - 평균 성적: {class_df['score'].mean():.2f}")
        print(f"  - 리더십 학생 수: {class_df[class_df['Leadership'] == 'yes'].shape[0]}명")
        print(f"  - 피아노 가능 학생 수: {class_df[class_df['Piano'] == 'yes'].shape[0]}명")
        print(f"  - 비등교 성향 학생 수: {class_df[class_df['비등교'] == 'yes'].shape[0]}명")
        print(f"  - 운동 선호 학생 수: {class_df[class_df['운동선호'] == 'yes'].shape[0]}명")

if __name__ == '__main__':
    solve_class_assignment()