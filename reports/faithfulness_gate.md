# Faithfulness quality gate

✅ PASSED: mean RAGAS faithfulness **0.787** on 20 questions (threshold 0.75); 20 answers judged.

Generated 2026-10-07 19:32 UTC by `python -m rag.evaluation.gate` (git afa1e3d). Answers: `Qwen/Qwen2.5-7B-Instruct-AWQ` from the production index's top 5 articles; judge: `Qwen/Qwen2.5-7B-Instruct-AWQ`. Arabic 0.642, English 0.933. Took 42 s.

Faithfulness = supported statements ÷ all statements in the answer (RAGAS): it checks that the answer sticks to the articles it was given, not that it is legally correct.

| Question | Relevant | Retrieved (top 5) | Faithfulness | Answer (start) |
|---|---|---|---|---|
| `q01-ar` إذا لم يوجد نص تشريعي يحكم النزاع، فعلى أي أساس يفصل القاضي فيه؟ | 1 | 1, 24, 838, 23, 200 | 1.00 | إذا لم يوجد نص تشريعي يحكم النزاع، فسوف يفصل القاضي فيه وفقًا للعرف، ثم إذا لم يوجد عرف، فبالأساس المبادئ الإسلامية، ثم  |
| `q01-en` If no statute covers a dispute, what should an Egyptian judge base the | 1 | 25, 23, 10, 28, 1 | 1.00 | If no statute covers a dispute, the Egyptian judge should base the decision on the following:  - According to Article 1( |
| `q03-ar` ما السن التي يبلغ فيها الشخص الرشد فيصبح كامل الأهلية لمباشرة حقوقه ال | 44 | 44, 46, 45, 112, 48 | 0.00 | الشخص يبلغ الرشد عند تمام двад十多年完成时，应使用阿拉伯数字，因此答案是：  الشخص يبلغ الرشد عند تمام двадى عشر سنة ميلادية كاملة، وذلك وفقًا  |
| `q03-en` How old must someone be to have full capacity to exercise their civil  | 44 | 44, 45, 46, 48, 112 | 1.00 | To have full capacity to exercise their civil rights, a person must attain majority, which is fixed at twenty-one years  |
| `q04-ar` في أي لحظة يتم إبرام العقد بين طرفين؟ | 89 | 89, 97, 101, 94, 106 | 1.00 | يتم إبرام العقد بمجرد أن يتبادل طرفان التعبير عن إرادتين متطابقتين، مع مراعاة ما يقرره القانون فوق ذلك من أوضاع معينة لا |
| `q04-en` At what point is a contract concluded between two people? | 89 | 97, 89, 543, 694, 95 | 1.00 | A contract is concluded between two people when they exchange concordant intentions, subject to any special formalities  |
| `q08-ar` طرأت أزمة عامة لم يكن من الممكن توقعها فجعلت تنفيذ عقدي مرهقاً جداً وإ | 147 | 147, 117, 215, 217, 213 | 0.50 | نعم، يمكن للمحكمة التدخل في هذا الحالة. وفقًا لـ[Article 147]، إذا طرأت حوادث استثنائية عامة لم يكن في السила توقعها وتر |
| `q08-en` An unforeseeable general crisis made my contract ruinously expensive t | 147 | 147, 213, 214, 217, 695 | 1.00 | Yes, a court can help in such a situation. According to Article 147:  "When, however, as a result of exceptional and unp |
| `q11-ar` خلال أي مدة يجب على المضرور رفع دعوى التعويض عن العمل غير المشروع؟ | 172 | 172, 180, 187, 698, 197 | 1.00 | يجب على المضرور رفع دعوى التعويض عن العمل غير المشروع خلال ثلاث سنوات من اليوم الذي علم فيه بحدوث الإضرار وبهوية الشخص ا |
| `q11-en` How long does a victim have to sue for compensation for an unlawful ac | 172 | 172, 180, 170, 197, 50 | 1.00 | A victim has up to three years from the date they knew of the injury and the person responsible to sue for compensation  |
| `q17-ar` هل يضمن البائع العيب الخفي في المبيع حتى لو كان يجهل وجوده؟ | 447 | 453, 449, 447, 439, 448 | 0.75 | نعم، يضمن البائع العيب الخفي في المبيع حتى لو كان يجهل وجوده، وذلك وفقًا لنص [Article 447]. حيث ينص القانون على أن البائ |
| `q17-en` Is a seller liable for a hidden defect in the goods even if he did not | 447 | 447, 449, 448, 450, 453 | 0.67 | Yes, the seller is liable for hidden defects in the goods even if he did not know about them. According to [Article 447] |
| `q19-ar` هل تصح هبة مال لن يملكه الواهب إلا في المستقبل؟ | 492 | 492, 494, 490, 489, 487 | 0.00 | هبة مال لن يملكه الواهب إلا في المستقبل باطلة حسب [Article 492]. |
| `q19-en` Can a person give away, as a gift, property they will only own in the  | 492 | 492, 486, 491, 493, 489 | 1.00 | A person cannot give away property they will only own in the future as a valid gift. According to Article 492, a gift of |
| `q22-ar` توفي المستأجر، فهل ينتهي عقد الإيجار تلقائياً؟ | 601 | 601, 598, 602, 626, 600 | 1.00 | لا ينتهي عقد الإيجار تلقائياً بوفاة المستأجر، بل يمكن لأرثته أن يطالبوا بانتهاء العقد إذا أثبتوا أن الظروف أصبحت أصعب با |
| `q22-en` My tenant died. Does the lease end automatically? | 601 | 601, 602, 598, 626, 600 | 1.00 | The lease does not automatically end upon the death of the tenant. According to [Article 601], a contract of lease is no |
| `q26-ar` ما المدة اللازمة لاكتساب ملكية عقار بالتقادم عن طريق الحيازة؟ | 968, 969 | 969, 970, 968, 975, 973 | 0.67 | المدة اللازمة لاكتساب ملكية عقار بالتقادم عن طريق الحيازة هي خمس سنوات،前提是您提供的文章中并没有直接提到关于不动产通过占有获得所有权的具体年限。根据《埃及民事法典》中的 |
| `q26-en` How long must someone possess land to become its owner by prescription | 968, 969 | 969, 968, 972, 973, 974 | 1.00 | To become the owner of land through acquisitive prescription, someone must possess the land continuously for five years, |
| `q27-ar` ما الذي يمنحه الرهن الرسمي للدائن؟ | 1030 | 1030, 1031, 1082, 1034, 1032 | 0.50 | الرهن الرسمي يمنح الدائن حقاً عيناً على العقار المرهون، بحيث يكون له الأولوية في الاستيفاء من ثمن هذا العقار مهما كانت ي |
| `q27-en` What does an official mortgage give the creditor? | 1030 | 1030, 1063, 1061, 1060, 1070 | 0.67 | An official mortgage gives the creditor a real right over an immovable property that is specifically designated for the  |
