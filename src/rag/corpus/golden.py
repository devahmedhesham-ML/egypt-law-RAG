"""Reference articles transcribed from the source PDF (43, 44, 88, 89 from page screenshots; 492 from p. 64).

The build compares its output against these; a mismatch means the extractor regressed.
"""

GOLDEN: dict[int, dict] = {
    43: {
        "text_en": (
            "A special domicile may be elected for the performance of a specific legal act.\n"
            "The election of domicile must be evidenced by writing.\n"
            "A domicile elected for the performance of a legal act shall be deemed to be the domicile in so far "
            "as all matters relating to such act are concerned, including the procedure for enforcement by legal "
            "means unless the election of domicile is expressly limited to certain special acts, excluding others."
        ),
        "text_ar": (
            "(١) يجوز اتخاذ موطن مختار لتنفيذ عمل قانوني معين.\n"
            "(٢) ولا يجوز إثبات وجود الموطن المختار إلا بالكتابة.\n"
            "(٣) والموطن المختار لتنفيذ عمل قانوني يكون هو الموطن بالنسبة إلى كل ما يتعلق بهذا العمل، بما فى ذلك "
            "إجراءات التنفيذ الجبري، إلا إذا اشترط صراحة قصر هذا الموطن على أعمال دون أخرى."
        ),
        "section_number": 2,
    },
    44: {
        "text_en": (
            "All persons attaining majority in possession of their mental faculties and not under legal "
            "disability, have full legal capacity to exercise their civil rights.\n"
            "The majority of a person is fixed at twenty one years completed in accordance with the Gregorian calendar."
        ),
        "text_ar": (
            "(١) كل شخص بلغ سن الرشد متمتعا بقواه العقلية، ولم يحجر عليه، يكون كامل الأهلية لمباشرة حقوقه المدنية.\n"
            "(٢) وسن الرشد هى إحدى وعشرون سنة ميلادية كاملة."
        ),
        "section_number": 2,
        "topic_number": 1,
    },
    88: {
        "text_en": (
            "Properties forming part of the public domain lose this status with the cessation of their allocation "
            "for public utility purposes.\n"
            "This cessation takes place by virtue of a law, or a decree, or in fact, or if the object of public "
            "utility for which they were allocated comes to an end."
        ),
        "text_ar": (
            "تفقد الأموال العامة صفتها العامة بإنتهاء تخصيصها للمنفعة العامة. وينتهي التخصيص بمقتضى قانون أو مرسوم "
            "أو قرار من الوزير المختص أو بالفعل، أو بانتهاء الغرض الذي من أجله خصصت تلك الأموال للمنفعة العامة."
        ),
        "part_number": None,
        "book_number": None,
        "chapter_number": None,
        "section_number": 3,
        "section_title_en": "The Classification of Things and Property",
        "section_title_ar": "تقسيم الأشياء والأموال",
    },
    89: {
        "text_en": (
            "A contract is created, subject to any special formalities that may be required by law for its "
            "conclusion, from the moment that two persons have exchanged two concordant intentions."
        ),
        "text_ar": (
            "يتم العقد بمجرد أن يتبادل طرفان التعبير عن إرادتين متطابقتين، مع مراعاة ما يقرره القانون فوق ذلك من "
            "أوضاع معينة لانعقاد العقد."
        ),
        "part_number": 1,
        "part_title_en": "Obligations or Personal Rights",
        "part_title_ar": "الالتزامات أو الحقوق الشخصية",
        "book_number": 1,
        "book_title_en": "Obligations Generally",
        "book_title_ar": "الالتزامات بوجه عام",
        "chapter_number": 1,
        "chapter_title_en": "Sources of Obligations",
        "chapter_title_ar": "مصادر الالتزام",
        "section_number": 1,
        "section_title_en": "Contracts",
        "section_title_ar": "العقد",
        "topic_number": 1,
        "topic_title_en": "Elements of Contracts",
        "topic_title_ar": "أركان العقد",
        "subtopic_title_en": "Consent",
        "subtopic_title_ar": "الرضاء",
    },
    492: {
        "text_en": "A gift of future property is void.",
        "text_ar": "تقع هبة الأموال المستقبلة باطلة.",
        "part_number": 1,
        "book_number": 2,
        "section_number": 3,
        "section_title_en": "Gifts",
    },
}
