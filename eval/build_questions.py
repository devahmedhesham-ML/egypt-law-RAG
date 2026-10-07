"""Writes eval/questions.jsonl, the evaluation question set (edit the list here, then rerun).

Each topic is asked in Arabic and English (same `pair`), so retrieval can be compared across
languages. `relevant_articles` are the articles a correct answer rests on; `reference` is a short
correct answer for RAGAS (in the question's language). Questions paraphrase the articles instead of
copying their wording, so retrieval is tested on meaning, not shared words.

Status: ACCEPTED as the project's evaluation set by the project owner (2026-10-07). Written from the
article texts; not reviewed by a legal expert (see docs/evaluation-dataset.md, "Limits").

`ci: true` marks the 20 questions of the CI quality gate: 10 topics in both languages, covering every book and
both question types (factual: q03 q04 q11 q19 q27; reasoning: q01 q08 q17 q22 q26).
"""

import json
from pathlib import Path

STATUS = "accepted"
CI_PAIRS = {"q01", "q03", "q04", "q08", "q11", "q17", "q19", "q22", "q26", "q27"}

# (pair, type, articles, en_question, en_reference, ar_question, ar_reference)
TOPICS = [
    ("q01", "reasoning", [1],
     "If no statute covers a dispute, what should an Egyptian judge base the decision on?",
     "On custom; failing custom, on the principles of Islamic law; failing those, on natural law and the rules of equity [Article 1].",
     "إذا لم يوجد نص تشريعي يحكم النزاع، فعلى أي أساس يفصل القاضي فيه؟",
     "يحكم القاضي بمقتضى العرف، فإن لم يوجد فبمبادئ الشريعة الإسلامية، فإن لم توجد فبمبادئ القانون الطبيعي وقواعد العدالة [المادة 1]."),
    ("q02", "factual", [29],
     "When does a person's legal personality start and when does it end?",
     "It starts when the child is born alive and ends at death; the law sets the rights of an unborn child [Article 29].",
     "متى تبدأ الشخصية القانونية للإنسان ومتى تنتهي؟",
     "تبدأ بتمام ولادته حياً وتنتهي بموته، والقانون يحدد حقوق الحمل المستكن [المادة 29]."),
    ("q03", "factual", [44],
     "How old must someone be to have full capacity to exercise their civil rights?",
     "Twenty-one full Gregorian years, for a person of sound mind who is not under interdiction [Article 44].",
     "ما السن التي يبلغ فيها الشخص الرشد فيصبح كامل الأهلية لمباشرة حقوقه المدنية؟",
     "إحدى وعشرون سنة ميلادية كاملة، متى كان متمتعاً بقواه العقلية ولم يحجر عليه [المادة 44]."),
    ("q04", "factual", [89],
     "At what point is a contract concluded between two people?",
     "As soon as the two parties exchange two matching expressions of will, subject to any form the law requires [Article 89].",
     "في أي لحظة يتم إبرام العقد بين طرفين؟",
     "يتم العقد بمجرد أن يتبادل الطرفان التعبير عن إرادتين متطابقتين، مع مراعاة ما يقرره القانون من أوضاع معينة لانعقاده [المادة 89]."),
    ("q05", "reasoning", [120],
     "I signed a contract while mistaken about something essential. Can I have it annulled?",
     "Yes, if the other party made the same mistake, knew of it, or could easily have noticed it [Article 120].",
     "وقعت في غلط جوهري عند التعاقد، فهل يحق لي طلب إبطال العقد؟",
     "نعم، إذا كان المتعاقد الآخر قد وقع في نفس الغلط أو كان على علم به أو كان من السهل عليه أن يتبينه [المادة 120]."),
    ("q06", "reasoning", [125],
     "Does deliberately keeping quiet about an important fact during negotiations count as fraud that lets the other side void the contract?",
     "Yes, intentional silence is fraud if it is proved the other party would not have contracted had they known the fact [Article 125].",
     "هل يعد كتمان أحد الطرفين عمداً لواقعة مهمة أثناء التعاقد تدليساً يبرر إبطال العقد؟",
     "نعم، يعتبر السكوت العمدي تدليساً إذا ثبت أن المتعاقد الآخر ما كان ليبرم العقد لو علم بتلك الواقعة [المادة 125]."),
    ("q07", "reasoning", [129],
     "Someone took advantage of my obvious recklessness to make me sign a grossly one-sided contract. What can I do, and how long do I have?",
     "Ask the judge to annul the contract or reduce your obligations; the claim must be filed within one year of the contract, and in a contract for value the other side can avoid annulment by offering adequate compensation [Article 129].",
     "استغل الطرف الآخر طيشي البيّن فأبرمت عقداً غير متكافئ إطلاقاً، فما الذي يمكنني فعله وما المدة المتاحة لي؟",
     "يجوز للقاضي بناءً على طلبك أن يبطل العقد أو ينقص التزاماتك، ويجب رفع الدعوى خلال سنة من تاريخ العقد، ويجوز للطرف الآخر في عقود المعاوضة أن يتوقى الإبطال بعرض ما يراه القاضي كافياً لرفع الغبن [المادة 129]."),
    ("q08", "reasoning", [147],
     "An unforeseeable general crisis made my contract ruinously expensive to perform, though not impossible. Can a court help?",
     "Yes, the judge may reduce the overly burdensome obligation to a reasonable level after balancing both parties' interests; any agreement excluding this is void [Article 147].",
     "طرأت أزمة عامة لم يكن من الممكن توقعها فجعلت تنفيذ عقدي مرهقاً جداً وإن لم يصبح مستحيلاً، فهل يمكن للمحكمة التدخل؟",
     "نعم، يجوز للقاضي بعد الموازنة بين مصلحة الطرفين أن يرد الالتزام المرهق إلى الحد المعقول، ويقع باطلاً كل اتفاق على خلاف ذلك [المادة 147]."),
    ("q09", "reasoning", [157],
     "The other party to my bilateral contract has not performed. What are my options?",
     "After a formal notice, you may demand performance or rescission, with damages in either case; the judge may give the debtor more time or refuse rescission if the unperformed part is minor [Article 157].",
     "لم ينفذ الطرف الآخر التزامه في عقد ملزم للجانبين، فما الخيارات المتاحة لي؟",
     "بعد إعذار المدين يجوز لك المطالبة بتنفيذ العقد أو بفسخه مع التعويض في الحالتين، وللقاضي أن يمنح المدين أجلاً أو يرفض الفسخ إذا كان ما لم يوف به قليل الأهمية [المادة 157]."),
    ("q10", "factual", [163],
     "Is a person who harms someone else through their own fault obliged to compensate them?",
     "Yes, every fault that causes harm to another obliges the person who committed it to compensate [Article 163].",
     "هل يلتزم من تسبب بخطئه في إلحاق ضرر بغيره بتعويضه؟",
     "نعم، كل خطأ سبب ضرراً للغير يلزم من ارتكبه بالتعويض [المادة 163]."),
    ("q11", "factual", [172],
     "How long does a victim have to sue for compensation for an unlawful act?",
     "Three years from when the victim learned of the harm and who caused it, and fifteen years from the act in any case; if the act is also a crime, not before the criminal action lapses [Article 172].",
     "خلال أي مدة يجب على المضرور رفع دعوى التعويض عن العمل غير المشروع؟",
     "تسقط بمضي ثلاث سنوات من يوم علمه بالضرر وبالشخص المسؤول عنه، وفي كل حال بمضي خمس عشرة سنة من يوم وقوع العمل، وإذا كانت ناشئة عن جريمة فلا تسقط إلا بسقوط الدعوى الجنائية [المادة 172]."),
    ("q12", "reasoning", [179],
     "Someone gained money at my expense without any legal justification. Must they pay me back?",
     "Yes, they must compensate your loss up to the amount of their gain, even if they lack discretion and even if the gain later disappears [Article 179].",
     "أثرى شخص على حسابي دون سبب قانوني، فهل يلتزم بتعويضي؟",
     "نعم، يلتزم في حدود ما أثرى به بتعويضك عما لحقك من خسارة، ولو كان غير مميز، ويبقى ملتزماً ولو زال إثراؤه فيما بعد [المادة 179]."),
    ("q13", "factual", [226],
     "What interest does a debtor owe for being late in paying a known sum of money?",
     "Four percent in civil matters and five percent in commercial matters, from the date of the court claim unless another date applies [Article 226].",
     "ما سعر الفوائد التي يلتزم بها المدين إذا تأخر في سداد مبلغ من النقود معلوم المقدار؟",
     "أربعة في المائة في المسائل المدنية وخمسة في المائة في المسائل التجارية، وتسري من تاريخ المطالبة القضائية ما لم يحدد الاتفاق أو العرف التجاري تاريخاً آخر [المادة 226]."),
    ("q14", "reasoning", [227],
     "Can a lender and borrower agree on ten percent annual interest?",
     "No, agreed interest may not exceed seven percent; a higher rate is reduced to seven percent and any excess paid must be refunded [Article 227].",
     "هل يجوز الاتفاق على فائدة سنوية قدرها عشرة في المائة على القرض؟",
     "لا، لا يجوز أن يزيد سعر الفائدة المتفق عليه على سبعة في المائة، فإن زاد خفض إليها ووجب رد ما دفع زائداً [المادة 227]."),
    ("q15", "factual", [374],
     "What is the ordinary limitation period for obligations?",
     "Fifteen years, except where the law provides otherwise [Article 374].",
     "ما مدة التقادم العادية التي يسقط بها الالتزام؟",
     "خمس عشرة سنة، فيما عدا الحالات التي ورد عنها نص خاص في القانون [المادة 374]."),
    ("q16", "factual", [418],
     "How does the Civil Code define a contract of sale?",
     "A contract by which the seller undertakes to transfer ownership of a thing or another property right to the buyer in return for a price in money [Article 418].",
     "كيف يعرّف القانون المدني عقد البيع؟",
     "البيع عقد يلتزم به البائع أن ينقل للمشتري ملكية شيء أو حقاً مالياً آخر في مقابل ثمن نقدي [المادة 418]."),
    ("q17", "reasoning", [447],
     "Is a seller liable for a hidden defect in the goods even if he did not know about it?",
     "Yes, the seller is liable for defects that reduce the thing's value or usefulness even if he was unaware of them [Article 447].",
     "هل يضمن البائع العيب الخفي في المبيع حتى لو كان يجهل وجوده؟",
     "نعم، يضمن البائع العيب الذي ينقص من قيمة المبيع أو من نفعه ولو لم يكن عالماً بوجوده [المادة 447]."),
    ("q18", "reasoning", [452],
     "I found a defect in something I bought two years after it was delivered. Can I still claim under the warranty?",
     "Normally no: the warranty claim lapses one year after delivery, unless the seller agreed to a longer period or fraudulently concealed the defect [Article 452].",
     "اكتشفت عيباً في شيء اشتريته بعد سنتين من تسلمه، فهل ما زال بإمكاني الرجوع على البائع بضمان العيب؟",
     "الأصل لا، إذ تسقط دعوى الضمان بمضي سنة من وقت التسليم، إلا إذا قبل البائع الضمان لمدة أطول أو ثبت أنه تعمد إخفاء العيب غشاً منه [المادة 452]."),
    ("q19", "factual", [492],
     "Can a person give away, as a gift, property they will only own in the future?",
     "No, a gift of future property is void [Article 492].",
     "هل تصح هبة مال لن يملكه الواهب إلا في المستقبل؟",
     "لا، هبة الأموال المستقبلة تقع باطلة [المادة 492]."),
    ("q20", "reasoning", [502],
     "Can a husband take back a gift he gave to his wife?",
     "No, a gift between spouses is one of the obstacles that prevent revoking a gift [Article 502(d)].",
     "هل يجوز للزوج أن يسترد هبة وهبها لزوجته؟",
     "لا، الهبة بين الزوجين من موانع الرجوع في الهبة [المادة 502 (د)]."),
    ("q21", "factual", [558],
     "What is a lease under the Civil Code?",
     "A contract by which the lessor undertakes to let the lessee use a specific thing for a set period in return for a fixed rent [Article 558].",
     "ما المقصود بعقد الإيجار في القانون المدني؟",
     "عقد يلتزم المؤجر بمقتضاه أن يمكن المستأجر من الانتفاع بشيء معين مدة معينة لقاء أجر معلوم [المادة 558]."),
    ("q22", "reasoning", [601],
     "My tenant died. Does the lease end automatically?",
     "No, a lease does not end with the death of the lessor or the lessee; the lessee's heirs may ask to end it within six months if it has become too heavy for them or exceeds their needs [Article 601].",
     "توفي المستأجر، فهل ينتهي عقد الإيجار تلقائياً؟",
     "لا، لا ينتهي الإيجار بموت المؤجر ولا بموت المستأجر، ولورثة المستأجر طلب إنهائه خلال ستة أشهر إذا أصبحت أعباؤه أثقل من مواردهم أو زاد عن حاجتهم [المادة 601]."),
    ("q23", "reasoning", [739],
     "I lost money on a bet. Is the bet enforceable, and can I get the money back?",
     "Gambling and betting agreements are void; the loser may recover what he paid within three years of paying, despite any contrary agreement [Article 739].",
     "خسرت مالاً في رهان، فهل الرهان ملزم، وهل يمكنني استرداد ما دفعته؟",
     "يقع باطلاً كل اتفاق خاص بمقامرة أو رهان، ولمن خسر أن يسترد ما دفعه خلال ثلاث سنوات من وقت الدفع ولو كان هناك اتفاق يقضي بغير ذلك [المادة 739]."),
    ("q24", "factual", [772],
     "What does a guarantor promise under a contract of suretyship?",
     "To perform the debtor's obligation towards the creditor if the debtor fails to do so [Article 772].",
     "بماذا يتعهد الكفيل في عقد الكفالة؟",
     "يتعهد للدائن بأن يفي بالتزام المدين إذا لم يف به المدين نفسه [المادة 772]."),
    ("q25", "factual", [802],
     "What rights does the owner of a thing have over it?",
     "The owner alone, within the limits of the law, may use it, exploit it and dispose of it [Article 802].",
     "ما الحقوق التي يتمتع بها مالك الشيء عليه؟",
     "للمالك وحده، في حدود القانون، حق استعمال الشيء واستغلاله والتصرف فيه [المادة 802]."),
    ("q26", "reasoning", [968, 969],
     "How long must someone possess land to become its owner by prescription?",
     "Fifteen years of uninterrupted possession; five years if the possessor holds an immovable in good faith under a registered just title [Articles 968 and 969].",
     "ما المدة اللازمة لاكتساب ملكية عقار بالتقادم عن طريق الحيازة؟",
     "خمس عشرة سنة من الحيازة المستمرة، وتكون المدة خمس سنوات إذا اقترنت الحيازة بحسن النية وسبب صحيح مسجل [المادتان 968 و969]."),
    ("q27", "factual", [1030],
     "What does an official mortgage give the creditor?",
     "A real right over an immovable set aside for the debt, giving priority over ordinary and later creditors in being paid from its price, whoever holds it [Article 1030].",
     "ما الذي يمنحه الرهن الرسمي للدائن؟",
     "يكسبه حقاً عينياً على عقار مخصص لوفاء دينه يتقدم بمقتضاه على الدائنين العاديين والتالين له في المرتبة في استيفاء حقه من ثمنه في أي يد يكون [المادة 1030]."),
    ("q28", "reasoning", [1130],
     "Can a creditor and debtor create a privileged right simply by agreeing on it?",
     "No, a privilege is a priority the law grants to a right because of its nature, and no right is privileged except by a provision of law [Article 1130].",
     "هل يستطيع الدائن والمدين إنشاء حق امتياز بمجرد اتفاقهما عليه؟",
     "لا، الامتياز أولوية يقررها القانون لحق معين مراعاة لصفته، ولا يكون للحق امتياز إلا بمقتضى نص في القانون [المادة 1130]."),
]

# (id, lang, type, articles, question, reference)
SINGLES = [
    ("q29-ar", "ar", "by_number", [60], "ماذا تنص المادة 60 من القانون المدني؟",
     "المادة 60 ملغاة، ضمن المواد من 54 إلى 80 التي ألغيت بالقرار الجمهوري بالقانون رقم 348 لسنة 1956 [المادة 60]."),
    ("q30-en", "en", "by_number", [505], "What does Article 505 say?",
     "It defines partnership: a contract by which two or more persons contribute property or work to a financial undertaking to share its profits or losses [Article 505]."),
    ("q31-en", "en", "out_of_scope", [], "What is the penalty for theft?",
     "The Civil Code does not set criminal penalties; the answer should say the provided articles do not cover it."),
    ("q32-en", "en", "out_of_scope", [], "How many days of paid annual leave is an employee entitled to?",
     "Annual leave is governed by labour law, not the Civil Code; the answer should decline."),
    ("q33-ar", "ar", "out_of_scope", [], "ما عقوبة تزوير المحررات الرسمية؟",
     "القانون المدني لا يتضمن عقوبات جنائية، ويجب أن توضح الإجابة أن المواد المقدمة لا تتناول ذلك."),
    ("q34-ar", "ar", "out_of_scope", [], "ما شروط الترشح لعضوية مجلس النواب؟",
     "هذه المسألة ينظمها الدستور وقوانين الانتخاب لا القانون المدني، ويجب أن تعتذر الإجابة."),
]


def rows() -> list[dict]:
    out = []
    for pair, kind, articles, en_q, en_ref, ar_q, ar_ref in TOPICS:
        for lang, question, reference in (("ar", ar_q, ar_ref), ("en", en_q, en_ref)):
            out.append({"id": f"{pair}-{lang}", "pair": pair, "lang": lang, "type": kind, "question": question,
                        "relevant_articles": articles, "reference": reference, "status": STATUS,
                        "ci": pair in CI_PAIRS})
    for qid, lang, kind, articles, question, reference in SINGLES:
        out.append({"id": qid, "pair": qid.split("-")[0], "lang": lang, "type": kind, "question": question,
                    "relevant_articles": articles, "reference": reference, "status": STATUS, "ci": False})
    return out


if __name__ == "__main__":
    path = Path(__file__).with_name("questions.jsonl")
    data = rows()
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
    print(f"wrote {len(data)} questions to {path}")
