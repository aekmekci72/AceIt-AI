import React, { useState, useEffect } from 'react';
import Navbar from './Navbar';
import axios from 'axios';

const QuizComponent = () => {
    const chapterId = localStorage.getItem('currentSection');
    const collName = localStorage.getItem('collectionName');
    const chapterName = localStorage.getItem('currentSectionName');
    const collectionId = localStorage.getItem('currentCollection');
    const [quizSubmitted, setQuizSubmitted] = useState(false);
    const [userAnswers, setUserAnswers] = useState({});
    const [correctAnswers, setCorrectAnswers] = useState({});
    const [showResults, setShowResults] = useState(false);
    const [questions, setQuestions] = useState([]);
    const [numTFQuestions, setNumTFQuestions] = useState(5);
    const [numFRQQuestions, setNumFRQQuestions] = useState(5);
    const [questionSource, setQuestionSource] = useState(''); // 'ai' or 'flashcards'
    const [numFlashcardQuestions, setNumFlashcardQuestions] = useState(5);
    const [startTime, setStartTime] = useState(null);
    const [endTime, setEndTime] = useState(null);
    const [totalTimeSpent, setTotalTimeSpent] = useState(0);
    const [score, setScore] = useState(0);
    const [totalQuestions, setTotalQuestions] = useState(0);
    const [questionsGenerated, setQuestionsGenerated] = useState(false); // Track if questions are generated

    useEffect(() => {
        let startTime = null;
        let endTime = null;

        const handleBeforeUnload = () => {
            endTime = new Date().getTime();
            if (startTime && endTime) {
                const timeSpent = endTime - startTime;
                setTotalTimeSpent(timeSpent);
                try {
                    axios.post('http://localhost:5000/time_spent', {
                        collection_id: collectionId,
                        section_id: chapterId,
                        total_time_spent: timeSpent,
                    });
                } catch (error) {
                    console.error('Error updating time spent:', error);
                }
            }
        };

        const handleUnload = () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
            window.removeEventListener('unload', handleUnload);
            endTime = new Date().getTime();
            if (startTime && endTime) {
                const timeSpent = endTime - startTime;
                try {
                    axios.post('http://localhost:5000/time_spent', {
                        collection_id: collectionId,
                        section_id: chapterId,
                        total_time_spent: timeSpent,
                    });
                } catch (error) {
                    console.error('Error updating time spent:', error);
                }
            }
        };

        window.addEventListener('beforeunload', handleBeforeUnload);
        window.addEventListener('unload', handleUnload);

        startTime = new Date().getTime();

        return () => {
            handleBeforeUnload();
        };
    }, [chapterId, collectionId]);

    const fetchQuestions = async () => {
        try {
            const sectionId = localStorage.getItem('currentSection');
            if (questionSource === 'ai') {
                const tfResponse = await axios.post(
                    `http://localhost:5000/generate_tf_questions`,
                    {
                        section_id: sectionId,
                        num_questions: numTFQuestions
                    },
                    {
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    }
                );
                const frqResponse = await axios.post(
                    `http://localhost:5000/generate_frq`,
                    {
                        section_id: sectionId,
                        num_questions: numFRQQuestions
                    },
                    {
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    }
                );
                const mcqResponse = await axios.post(
                    `http://localhost:5000/generate_mcq`,
                    {
                        section_id: sectionId,
                        num_questions: numFRQQuestions
                    },
                    {
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    }
                );
                setQuestions([
                    ...tfResponse.data.map(q => ({ ...q, type: 'tf' })),
                    ...frqResponse.data.map(q => ({ ...q, type: 'frq' }))
                ]);
            } else {
                const flashcardResponse = await axios.post(
                    `http://localhost:5000/get_flashcards_frq`,
                    {
                        collection_id: localStorage.getItem('currentCollection'),
                        section_id: localStorage.getItem('currentSection'),
                        num_questions: numFlashcardQuestions
                    },
                    {
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    }
                );
                setQuestions(flashcardResponse.data.flashcards.map(q => ({ ...q, type: 'frq' })));
            }
            setQuestionsGenerated(true); // Mark questions as generated
        } catch (error) {
            console.error('Error fetching questions:', error);
        }
    };

    const handleAnswerChange = (e, index) => {
        setUserAnswers({
            ...userAnswers,
            [index]: e.target.value
        });
    };

    const handleSubmit = () => {
        let correct = {};
        let totalScore = 0;
        questions.forEach((q, index) => {
            if (q.type === 'tf') {
                correct[index] = q.answer.toLowerCase() === 'true';
                if (correct[index]) {
                    totalScore++;
                }
            } else {
                correct[index] = q.answer.toLowerCase();
                if (checkAnswer(userAnswers[index], q.answer, q.type)) {
                    totalScore++;
                }
            }
        });
        setCorrectAnswers(correct);
        setShowResults(true);
        setScore(totalScore);
        setTotalQuestions(questions.length);

        saveTestScore(totalScore, questions.length);
    };

    const resetQuiz = () => {
        setUserAnswers({});
        setCorrectAnswers({});
        setShowResults(false);
        setScore(0);
        setQuestions([]);
        setQuestionsGenerated(false); // Reset questions generated flag
        setQuestionSource(''); // Reset question source
    };

    const saveTestScore = async (score, totalQuestions) => {
        try {
            await axios.post('http://localhost:5000/save_test_score', {
                collection_id: collectionId,
                section_id: chapterId,
                percentage: ((score / totalQuestions) * 100).toFixed(2),
            });
        } catch (error) {
            console.error('Error saving test score:', error);
        }
    };

    const checkAnswer = (userAnswer, correctAnswer, type) => {
        if (type === 'tf') {
            return userAnswer.toLowerCase() === correctAnswer.toString().toLowerCase();
        } else {
            return userAnswer.toLowerCase().includes(correctAnswer.toLowerCase());
        }
    };

    return (
        <div className="bg-gray-900 text-gray-100 font-sans min-h-screen flex flex-col">
            <Navbar />
            <div className="container mx-auto px-4 py-8 flex flex-col items-center flex-grow overflow-auto">
                <h2 className="text-5xl font-bold text-center mb-12 text-cyan-400">Practice Test</h2>
                <div className="max-w-3xl w-full bg-gray-800 rounded-lg shadow-lg p-8 mb-8">
                    {!questionsGenerated && (
                        <div className="mb-6 flex justify-center space-x-6">
                            <label className="inline-flex items-center">
                                <input
                                    type="radio"
                                    className="form-radio text-cyan-600"
                                    name="questionSource"
                                    value="ai"
                                    checked={questionSource === 'ai'}
                                    onChange={() => setQuestionSource('ai')}
                                />
                                <span className="ml-2">AI Generated Questions</span>
                            </label>
                            <label className="inline-flex items-center">
                                <input
                                    type="radio"
                                    className="form-radio text-cyan-600"
                                    name="questionSource"
                                    value="flashcards"
                                    checked={questionSource === 'flashcards'}
                                    onChange={() => setQuestionSource('flashcards')}
                                />
                                <span className="ml-2">Flashcards</span>
                            </label>
                        </div>
                    )}
                    
                    {questionSource && !questionsGenerated && (
                        <div className="space-y-4">
                            {questionSource === 'ai' ? (
                                <>
                                    <div>
                                        <label className="block mb-2">Number of True/False Questions:</label>
                                        <input
                                            type="number"
                                            className="w-full bg-gray-700 text-white border border-gray-600 rounded-lg py-2 px-3 focus:outline-none focus:border-cyan-400"
                                            value={numTFQuestions}
                                            onChange={(e) => setNumTFQuestions(Number(e.target.value))}
                                        />
                                    </div>
                                    <div>
                                        <label className="block mb-2">Number of Free Response Questions:</label>
                                        <input
                                            type="number"
                                            className="w-full bg-gray-700 text-white border border-gray-600 rounded-lg py-2 px-3 focus:outline-none focus:border-cyan-400"
                                            value={numFRQQuestions}
                                            onChange={(e) => setNumFRQQuestions(Number(e.target.value))}
                                        />
                                    </div>
                                </>
                            ) : (
                                <div>
                                    <label className="block mb-2">Number of Flashcard Questions:</label>
                                    <input
                                        type="number"
                                        className="w-full bg-gray-700 text-white border border-gray-600 rounded-lg py-2 px-3 focus:outline-none focus:border-cyan-400"
                                        value={numFlashcardQuestions}
                                        onChange={(e) => setNumFlashcardQuestions(Number(e.target.value))}
                                    />
                                </div>
                            )}
                            <button
                                className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-2 px-4 rounded-lg"
                                onClick={fetchQuestions}
                            >
                                Get Questions
                            </button>
                        </div>
                    )}

                    {questions.length > 0 && (
                        <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }} className="space-y-6">
                            {questions.map((question, index) => (
                                <div key={index} className="bg-gray-700 p-4 rounded-lg shadow-md">
                                    <p className="text-lg font-semibold">{index + 1}. {question.question}</p>
                                    {question.type === 'tf' ? (
                                        <div className="mt-2">
                                            <label className="inline-flex items-center">
                                                <input
                                                    type="radio"
                                                    className="form-radio text-cyan-600"
                                                    name={`question_${index}`}
                                                    value="true"
                                                    checked={userAnswers[index] === 'true'}
                                                    onChange={(e) => handleAnswerChange(e, index)}
                                                />
                                                <span className="ml-2">True</span>
                                            </label>
                                            <label className="inline-flex items-center ml-6">
                                                <input
                                                    type="radio"
                                                    className="form-radio text-cyan-600"
                                                    name={`question_${index}`}
                                                    value="false"
                                                    checked={userAnswers[index] === 'false'}
                                                    onChange={(e) => handleAnswerChange(e, index)}
                                                />
                                                <span className="ml-2">False</span>
                                            </label>
                                        </div>
                                    ) : (
                                        <input
                                            type="text"
                                            className="w-full bg-gray-700 text-white border border-gray-600 rounded-lg py-2 px-3 focus:outline-none focus:border-cyan-400"
                                            value={userAnswers[index] || ''}
                                            onChange={(e) => handleAnswerChange(e, index)}
                                        />
                                    )}
                                </div>
                            ))}
                            <button
                                type="submit"
                                className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-2 px-4 rounded-lg"
                            >
                                Submit
                            </button>
                        </form>
                    )}

                    {showResults && (
                        <div className="mt-8">
                            <h3 className="text-2xl font-bold">Results</h3>
                            <p>Your score: {score} out of {totalQuestions}</p>
                            <div className="mt-4">
                                {questions.map((question, index) => (
                                    <div key={index} className="bg-gray-700 p-4 rounded-lg mt-2">
                                        <p className="text-lg font-semibold">{index + 1}. {question.question}</p>
                                        <p className="text-sm mt-2">Your answer: {userAnswers[index]}</p>
                                        <p className={`text-sm mt-2 ${correctAnswers[index] ? 'text-green-500' : 'text-red-500'}`}>
                                            Correct answer: {question.answer}
                                        </p>
                                    </div>
                                ))}
                            </div>
                            <button
                                className="mt-4 bg-gray-700 hover:bg-gray-800 text-white font-bold py-2 px-4 rounded-lg"
                                onClick={resetQuiz}
                            >
                                New Test
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default QuizComponent;
