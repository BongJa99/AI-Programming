import pandas as pd
from ortools.sat.python import cp_model
import numpy as np

# #############################################################################
# --- (변경) 설정 영역 ---
# 여기서 모든 것을 제어합니다.
# #############################################################################
CONFIG = {
    "CSV_PATH": "class_feature.csv",
    "NUM_CLASSES": 6,
    # 'yes'/'no' 또는 1/0 형태의 값을 가진, 균등 배정이 필요한 모든 컬럼 목록
    "BALANCE_YES_NO_COLUMNS": [
        'Leadership', 'Piano', '비등교', '운동선호', '노래', '댄스',
        '야구', '미술', '밴드', '축구', '코딩', '독서', '봉사', '연극'
    ],
    # 범주형 데이터(성별, 클럽 등) 중 균등 배정이 필요한 컬럼 목록
    "BALANCE_CATEGORY_COLUMNS": ['sex', '클럽']
}

# --- (변경) 최적화 가중치 설정 ---
# 각 패널티의 중요도를 여기서 조절합니다. 숫자가 클수록 더 중요하게 여깁니다.
WEIGHTS = {
    "PROPERTY_BALANCE": 10,  # 'yes'/'no' 특성들의 균등 배분 가중치
    "CATEGORY_BALANCE": 10,  # '성별', '클럽' 등 카테고리 균등 배분 가중치
    "SCORE_BALANCE": 1,      # 성적 편차 가중치
    "LAST_YEAR_OVERLAP": 50  # 작년 동급생 중복 최소화 가중치
}
# #############################################################################


def print_summary(df, config):
    """
    (개선) 배정 결과를 하나의 통합된 표로 요약하여 한눈에 보기 쉽게 출력합니다.
    """
    print("\n" + "="*60)
    print(" " * 18 + "반 배정 통합 요약 리포트")
    print("="*60)

    assigned_class_col = '배정된반'
    class_labels = sorted(df[assigned_class_col].unique())

    # 요약 정보를 저장할 데이터프레임 생성
    summary_df = pd.DataFrame(columns=class_labels)
    summary_df.columns.name = '배정된 반'

    # --- 1. 총 인원 ---
    summary_df.loc['총원'] = df[assigned_class_col].value_counts().sort_index()

    # --- 2. 'yes' 특성 분포 ---
    summary_df.loc['──────────'] = ''
    summary_df.loc['[특성 학생수]'] = ''
    for col in config["BALANCE_YES_NO_COLUMNS"]:
        if col in df.columns and (df[col] == 'yes').any():
            counts = df[df[col] == 'yes'].groupby(assigned_class_col).size().reindex(class_labels, fill_value=0)
            summary_df.loc[col] = counts

    # --- 3. 카테고리 특성 분포 ---
    for col in config["BALANCE_CATEGORY_COLUMNS"]:
        if col in df.columns:
            summary_df.loc['──────────'] = ''
            summary_df.loc[f'[{col} 분포]'] = ''
            
            crosstab_df = pd.crosstab(df[col], df[assigned_class_col])
            summary_df = pd.concat([summary_df, crosstab_df])

    # --- 4. 성적 통계 ---
    if 'score' in df.columns:
        summary_df.loc['──────────'] = ''
        summary_df.loc['[성적 통계]'] = ''
        summary_df.loc['평균 점수'] = df.groupby(assigned_class_col)['score'].mean().round(2)
    
    # (수정) to_string()을 사용하여 출력 정렬을 보정합니다.
    print(summary_df.fillna('').to_string())
    print("\n" + "="*60)


def solve_class_assignment(config, weights):
    """
    학생 데이터를 CSV 파일에서 읽어와 최적의 반 배정 문제를 해결합니다.
    """
    try:
        df = pd.read_csv(config["CSV_PATH"], encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(config["CSV_PATH"], encoding='cp949')
    except FileNotFoundError:
        print(f"오류: '{config['CSV_PATH']}' 파일을 찾을 수 없습니다.")
        return

    # --- 1. 기본 데이터 설정 ---
    num_students = len(df)
    num_classes = config["NUM_CLASSES"]
    students = list(df['id'])
    classes = list(range(num_classes))
    
    base_size, remainder = divmod(num_students, num_classes)
    class_sizes = {i: base_size + 1 if i < remainder else base_size for i in classes}
        
    print(f"총 학생 수: {num_students}명")
    print(f"클래스별 정원: {class_sizes}")

    # --- 2. CP-SAT 모델 생성 ---
    model = cp_model.CpModel()
    assign = {(s, c): model.NewBoolVar(f'assign_s{s}_c{c}') for s in students for c in classes}

    # --- 3. 하드 제약 조건 (반드시 지켜야 할 규칙) ---
    for s in students:
        model.AddExactlyOne([assign[(s, c)] for c in classes])

    for c in classes:
        model.Add(sum(assign[(s, c)] for s in students) == class_sizes[c])
        
    for _, row in df[df['나쁜관계'].notna()].iterrows():
        s1, s2 = row['id'], int(row['나쁜관계'])
        if s2 in students:
            for c in classes:
                model.AddBoolOr([assign[(s1, c)].Not(), assign[(s2, c)].Not()])

    for _, row in df[(df['비등교'] == 'yes') & (df['좋은관계'].notna())].iterrows():
        s1, s2 = row['id'], int(row['좋은관계'])
        if s2 in students:
            for c in classes:
                model.Add(assign[(s1, c)] == assign[(s2, c)])

    if 'Leadership' in df.columns:
        leaders = df[df['Leadership'] == 'yes']['id'].tolist()
        if leaders:
            for c in classes:
                model.Add(sum(assign[(s, c)] for s in leaders) >= 1)

    # --- 4. 소프트 제약 조건 (최적화 목표) ---
    all_penalty_terms = []

    for col in config["BALANCE_YES_NO_COLUMNS"]:
        if col in df.columns:
            student_group = df[df[col] == 'yes']['id'].tolist()
            if student_group:
                counts = [sum(assign[(s, c)] for s in student_group) for c in classes]
                max_v, min_v = model.NewIntVar(0, len(student_group), ''), model.NewIntVar(0, len(student_group), '')
                model.AddMaxEquality(max_v, counts)
                model.AddMinEquality(min_v, counts)
                all_penalty_terms.append((max_v - min_v) * weights["PROPERTY_BALANCE"])

    for col in config["BALANCE_CATEGORY_COLUMNS"]:
        if col in df.columns:
            for category in df[col].dropna().unique():
                student_group = df[df[col] == category]['id'].tolist()
                if student_group:
                    counts = [sum(assign[(s, c)] for s in student_group) for c in classes]
                    max_v, min_v = model.NewIntVar(0, len(student_group), ''), model.NewIntVar(0, len(student_group), '')
                    model.AddMaxEquality(max_v, counts)
                    model.AddMinEquality(min_v, counts)
                    all_penalty_terms.append((max_v - min_v) * weights["CATEGORY_BALANCE"])

    if 'score' in df.columns:
        score_map = df.set_index('id')['score']
        score_sums = [sum(assign[(s, c)] * score_map[s] for s in students) for c in classes]
        max_s, min_s = model.NewIntVar(0, int(df['score'].sum()), ''), model.NewIntVar(0, int(df['score'].sum()), '')
        model.AddMaxEquality(max_s, score_sums)
        model.AddMinEquality(min_s, score_sums)
        all_penalty_terms.append((max_s - min_s) * weights["SCORE_BALANCE"])

    if '24년 학급' in df.columns:
        overlap_vars = []
        for _, group in df.groupby('24년 학급'):
            group_students = group['id'].tolist()
            for i in range(len(group_students)):
                for j in range(i + 1, len(group_students)):
                    s1, s2 = group_students[i], group_students[j]
                    for c in classes:
                        overlap = model.NewBoolVar('')
                        model.AddBoolAnd([assign[(s1, c)], assign[(s2, c)]]).OnlyEnforceIf(overlap)
                        overlap_vars.append(overlap)
        total_overlaps = sum(overlap_vars)
        all_penalty_terms.append(total_overlaps * weights["LAST_YEAR_OVERLAP"])

    model.Minimize(sum(all_penalty_terms))

    # --- 5. 솔버 실행 및 결과 출력 ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120.0
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n✅ 최적의 반 배정 결과를 찾았습니다!")
        df['배정된반'] = [next(c + 1 for c in classes if solver.Value(assign[(s, c)])) for s in students]
        
        print_summary(df, config)
        
        df.to_csv("result.csv", index=False, encoding='utf-8-sig')
        print("\n결과가 'result.csv' 파일로 저장되었습니다.")
    else:
        print("\n❌ 모든 제약 조건을 만족하는 해를 찾을 수 없습니다. (하드 제약 조건 충돌 가능성)")



if __name__ == '__main__':
    solve_class_assignment(CONFIG, WEIGHTS)