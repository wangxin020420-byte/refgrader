"""Serialize individually inspected F columns; no OCR, classifier, or model call.

Vectors and comments below were manually entered after viewing each original
image and its frozen historical dossier. Counts are offline bookkeeping only.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = '0000000101111111'
# ID, manually read F vector, category, original observations, historical issue.
ROWS = [
 ('35','0000000101111111','model_or_extraction_issue','原图4；F列前七0、第8格1、第9格0；后七格改前的0被划去，右侧1是最终值。','把删除痕和新1合成81/01，及把圆0读D；不存在空末格，不能把改写当额外数字。'),
 ('45','0000000101111111','model_or_extraction_issue','原图4；上七个圈0、第8竖1、第9圈0、下七竖1均按行填满。','将延伸至格线附近的1错对行并漏掉下半格，所谓缺失并非原图空白。'),
 ('48','0000000101111111','model_or_extraction_issue','原图4；F列16格均有笔画，首七0、0111行1、1000行0，最后七格1。','提取仅14项且把1读一、尾格判空；原图没有缺末三行。'),
 ('50','0000000101111111','model_or_extraction_issue','原图4；第三行是带回勾的0，与其他圈0同形；第8和第10至16行竖1，1000行0。','第三格0被读6，多出第17项并错行；竖1不是空白占位符，不能按乱码扣分。'),
 ('39','0000000101111111','model_or_extraction_issue','原图4；首七与第9格均小圈0，其余1；最后一格一个竖1，未写二。','把0读D、1读一或二后当语义错；原图按16输入行对齐即完整正确。'),
 ('54','0000000101111111','model_or_extraction_issue','原图4；第3格大圈、第5格略带钩，均为0；第8格1，第9格0，下七1。','圈0的O/6及手写1的一是提取错误；不能把17项的提取串当学生表有17格。'),
 ('04','0000000101111111','model_or_extraction_issue','原图4；F列各输入行均有值，1001、1010行都是1，下方七1完整。','历史串多插0及一，使第10格被误判0；需读该格手写1而非重排噪声序列。'),
 ('22','0000000101111111','model_or_extraction_issue','原图4；前七与1000行圆0，0111及1001至1111行均竖1，16格完整。','模型称一不能表示1而扣；原图实际竖1清楚，即便提取一也应核图，不机械按字面判非法。'),
 ('29','0000000101111111','model_or_extraction_issue','原图4；F列下七行每格各有1，末格位于表内而不是缺失。','提取只有13项，称后七没完成；下七个1均在原图，不能由短转录推断学生漏填。'),
 ('40','0000000101111111','model_or_extraction_issue','原图4；较斜的0和细1均按行可辨，末1111行的1在下边线以上。','抽取多出空第17项和一/异或解释，图中不存在对应空行或异或符号。'),
 ('53','0000000000000001','model_or_extraction_issue','原图2；F前15行均0，唯末1111行1；不是全表空白。','参考本有8个0，这些以及末1应计部分分；将整个映射视核心错误清零，抹掉9个正确格。'),
 ('32','0000000101111111','model_or_extraction_issue','原图4；7个0后1，再0，最后七个细竖1全部可见。','O与一是OCR形状替换，18项串是提取错位，不代表原表逻辑错误。'),
 ('38','0000000101111111','model_or_extraction_issue','原图4；0111及1001至1111行的0已划去，旁边新1为最终值；1000行未改0。','把划去0和1拼为81/a1/6.1等再判数值错，忽略修改符号及格子边界。'),
 ('03','0000000101111111','model_or_extraction_issue','原图4；F列齐全，1100行旧0划掉改1，其余1细而连贯，首七0及1000行0明确。','把该格划改拼为Q1并漏相邻细1，后按乱码判0；原图无缺失输出。'),
 ('09','0000000101111111','model_or_extraction_issue','原图4；上方小圈0多呈开口形，0111行长1、1000行圈0，下七格1。','小圈被读C/D/2，细1读一或空，不能以这些转录字符证明学生逻辑错误。'),
 ('47','0000000101111111','model_or_extraction_issue','原图4；前七0较像D，第8格竖1、第9圈0，下七1部分很淡但逐格存在。','D和missing源于笔形及淡墨提取；并非学生只填两三个正确格，原图16格完整。'),
 ('37','0000000101111111','model_or_extraction_issue','原图4；F列正确，右另画卡诺图并列最小项7、9、10、11、12、13、14、15。','提取17项使第8格错成0；卡诺图/最小项又交叉证明0111输出1及下七1。'),
 ('13','0000000101111111','model_or_extraction_issue','原图4；首七大圈0，第8格1、第9圈0，下七1清楚。','将圆0当字母O并额外插一格后称转折遗漏；原表0111行1并未漏。'),
 ('12','0000000001111111','model_or_extraction_issue','原图3.5；仅0111行误填0，其余15格符合参考，1001起七格1。','历史提取18项且把多个1判一，另有11/16乘4等于1的算术错；实际只有一个错误，不是多数未答。'),
 ('14','0000000101111111','model_or_extraction_issue','原图4；0110行0、0111行1、1000行0清楚，下方1001至1111各1。','模型串把0110/0111转折错位并多出一项，所谓第7行1不是原图该格值。'),
 ('26','0000000101111111','model_or_extraction_issue','原图4；上七0、第8格1、第9格0、下七细1，最后1110与1111都填1。','将下半细1标字迹模糊、甚至student_blank；不能因取证不可靠就清零完整可读表。'),
 ('41','0000000101111111','model_or_extraction_issue','原图4；上七圈0、0111行1、1000行0，下七1清楚无空行。','O/一及多余空项为提取产物，不应把只有三个数字的文本当实际学生作答。'),
 ('49','0000000101111111','model_or_extraction_issue','原图4；F填在格左侧，上七0、0111行1、1000行0，下七竖1。','输出靠格左仍属F列，不能错丢一行并将多个1作未答；第一问对应格仍齐全。'),
 ('21','0000000101111111','model_or_extraction_issue','原图4；1001行1在其格中部，下六1连续，0111为1、1000为0。','历史串在1000后重复插0使第10格错，后七1也被改一/空；按输入行定位并无这些错误。'),
 ('05','0000000101111111','model_or_extraction_issue','原图4；首七圈0及1000行圈0，0111和下七竖1；1101格虽细仍有1。','17项串额外0和空造成后半错位，不能将这些噪声当学生少答六格。'),
 ('25','0000000101111111','model_or_extraction_issue','原图4；F上七0较淡，0111行1、1000行0、后七1；右注A不赞成需BCD全赞成、A赞成需BCD至少一人。','14项提取漏格，右侧文字还支持关键转折，不应因D/一和短序列扣大半。'),
 ('34','0000000101111111','model_or_extraction_issue','原图4；F小圈0带钩，第8长1、第9圈0，下七1均存在。','所谓缺失和第8格0与原图相反；取证按行后不能只认4或8格正确。'),
 ('43','0000000101111111','model_or_extraction_issue','原图4；第7圈0靠下边线，第8长1跨近格线但属于0111，1000为0，下七斜1。','跨近格线的1被漏，后七斜1也判missing；原图并非下半空表。'),
 ('19','0000000101111111','model_or_extraction_issue','原图4；F各格明确，前七0、0111为1、1000为0，最后七1。','历史说后11111与11111发生偏差自相矛盾，且17项提取多一尾项，不能据此扣分。'),
 ('36','0000000001111111','model_or_extraction_issue','原图3.5；0111行0错误，1000行0及其他15格正确，末七格1。','抽取18项并重复多0后错误计数；实际与12一样只有0111一处错，教师部分分有依据。'),
 ('08','1111111010000000','label_supported','原图0；前七1、0111行0、1000行1、最后七0，16格恰好与参考相反。','模型总0与逐行核对一致；其理由粗称前半全0/后半全1不够准确，但不影响此份0正确。'),
 ('17','0000000101111111','label_supported','原图4；F列16格清楚，上七0、第8格1、第9格0，下七1；转录亦按整行列全。','三次4与原图及参考完全一致，无需以归一化not_comparable怀疑教师满分。'),
 ('56','0000000101111111','label_supported','原图4；F首七圈0、0111竖1、1000圈0、最后七竖1清楚。','历史三次4正确，但一属于提取转写而非原图真的写中文横一；17项串不应取代16格行定位。'),
]

def main():
    records = []
    for aid, vector, category, observed, issue in ROWS:
        assert len(vector) == 16 and set(vector) <= {'0', '1'}
        wrong = [i for i, (a, b) in enumerate(zip(vector, REFERENCE)) if a != b]
        evidence = [
            '按原图A/B/C/D从0000至1111的输入行读取F：' + ', '.join(vector) + '。',
            '与冻结参考逐格核对：正确' + str(16-len(wrong)) + '/16；不同输入：' + (', '.join(format(i, '04b') for i in wrong) or '无') + '。',
        ]
        summary = ('F列16格全对，教师满4有原图支持，历史低分来自读格或取证错误。' if not wrong and category != 'label_supported'
                   else '只有0111行一处错误，其余15格正确，教师3.5可支持部分分。' if aid in ('12','36')
                   else '前15格0、末格1，仍有9格正确，教师2有部分分支持，模型全零过严。' if aid == '53'
                   else 'F列逐格与参考相反，教师0有支持。' if aid == '08'
                   else 'F列16格正确，教师4与模型满分一致。')
        records.append(dict(question_id='ML_1', answer_id='ANS_ML_'+aid, category=category,
            summary=summary, teacher_issue='', model_issue=issue,
            original_image_observations=observed, evidence=evidence,
            uncertainties=(['原始总4无每错一格硬扣多少；正确格数不是擅设新的逐格计分标准。'] if aid in ('12','36','53') else []),
            verified_truth_table=dict(input_order='0000..1111', manually_read_F=vector,
                frozen_reference_F=REFERENCE, matching_cells=16-len(wrong),
                mismatching_inputs=[format(i,'04b') for i in wrong],
                method='manual_original_image_inspection_then_offline_count')))
    assert len(records) == len({x['answer_id'] for x in records}) == 33
    target = ROOT / 'batches/ML_1_001.json'
    if target.exists():
        raise FileExistsError('Preserve existing batch: '+str(target))
    target.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print('Recorded 33 individually inspected F columns; no experiment or API.')

if __name__ == '__main__':
    main()
