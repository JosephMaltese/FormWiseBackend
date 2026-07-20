import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from "react-router-dom";
import { Spinner } from '../components/ui/shadcn-io/spinner';
import Navbar from '../components/ui/navbar';
import { Trophy, Target, FileVideo, MessageSquare, Check } from 'lucide-react';
import supabase from '../lib/supabaseClient';
import { Button } from '../components/ui/button';
import { UserAuth } from '../context/AuthContext';

export default function Results() {
    const location = useLocation();
    const { videoFile, exercise } = location.state;
    const [loading, setLoading] = useState(true);
    const [score, setScore] = useState(null);
    const [url, setUrl] = useState(null);
    const [analysis, setAnalysis] = useState("");
    const [saved, setSaved] = useState(false);
    const navigate = useNavigate();
    const { session } = UserAuth();
    const [hasProcessed, setHasProcessed] = useState(false);

    const handleReturn = () => {
        navigate('/')
    }

    useEffect(() => {
        if (hasProcessed) return;
        const fetchResults = async () => {
            try {
                const formData = new FormData();
                formData.append("file", videoFile);
                formData.append("exercise", exercise);
                const res = await fetch("http://127.0.0.1:8000/analyze-video/", {
                    method: "POST",
                    body: formData,
                });
                if (!res.ok) {
                    throw new Error(`HTTP error! status: ${res.status}`);
                }
                setScore(res.headers.get("total_score"));

                const data = await res.blob();
                setUrl(URL.createObjectURL(data));
                setLoading(false);

                const summaryFormData = new FormData();
                summaryFormData.append("exercise", exercise);
                summaryFormData.append("rep_count", res.headers.get("rep_count"));
                summaryFormData.append()
                const summaryRes = await fetch("http://127.0.0.1:8000/ai-summary/", {
                    method: "GET",
                    body: summaryFormData,
                })
                const encodedAnalysis = res.headers.get("gpt_analysis");
                if (encodedAnalysis) {
                    try {
                        const binaryString = atob(encodedAnalysis);
                        const bytes = new Uint8Array(binaryString.length);
                        for (let i = 0; i < binaryString.length; i++) {
                            bytes[i] = binaryString.charCodeAt(i);
                        }
                        const decodedAnalysis = new TextDecoder().decode(bytes);
                        setAnalysis(decodedAnalysis);
                    } catch (error) {
                        console.error('Error decoding analysis:', error);
                        setAnalysis('Error decoding analysis');
                    }
                }

                // Convert blob data to Video File
                const processedVideoFile = new File([data], `processed-${videoFile.name}`, {
                    type: data.type
                });

                // Get the current user from the session if it exists
                if (!session || !session.user) {
                    console.error('User not authenticated');
                    // Handle not authenticated case
                    return;
                }
                const userId = session.user.id;

                // Store the video file in supabase storage bucket
                const { videoData, videoError } = await supabase.storage.from('videos').upload(`${userId}/${Date.now()}-processed-${videoFile.name}`, processedVideoFile);

                // save analysis in supabase database
                const { error } = await supabase.from('videos').insert({ file_url: `${userId}/${Date.now()}-processed-${videoFile.name}`, user_id: userId, exercise_type: exercise, score: score, analysis: analysis })
                setSaved(true);

                setHasProcessed(true);
            } catch (err) {
                console.error("Failed to fetch results from backend:", err);
            }
        };

        fetchResults();
    }, [])

    return (
        loading ? (
            <div className='flex flex-col justify-center items-center h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800'>
                    <h1 className='text-3xl md:text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-6'>
                        Analyzing Your Form
                    </h1>
                    <Spinner variant='ellipsis' className='text-blue-600 dark:text-blue-400' size={128}/>
            </div>
        ) : (
            <div className='h-screen flex flex-col dark:text-white bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800'>
                <Navbar/>
                
                {/* Header Section */}
                <div className='flex-shrink-0 pt-4 pb-2 px-4'>
                    <div className='text-center'>
                        <h1 className='text-2xl md:text-3xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent'>
                            Form Analysis Results
                        </h1>
                        <p className='text-sm text-gray-600 dark:text-gray-300'>
                            Exercise: <span className='font-semibold text-blue-600 dark:text-blue-400'>{exercise}</span>
                        </p>
                        {saved && <p className='flex items-center justify-center gap-2 mt-2'>Your results have been saved <Check className='h-4 w-4 text-green-500' /></p>}
                    </div>

                    {/* Score Display */}
                    {score && (
                        <div className='max-w-xs mx-auto mt-3'>
                            <div className='flex flex-row bg-gradient-to-r justify-center items-center from-green-500 to-emerald-600 rounded-lg p-1 text-white shadow-lg'>
                                    <Trophy className='h-7 w-7 mr-2' />
                                    <h2 className='text-xl font-bold mr-4'>Score</h2>
                                    <span className='text-3xl font-bold mr-1'>{Math.round(score)}</span>
                                    <span className='text-sm ml-1'>/ 100</span>
                            </div>
                        </div>
                    )}
                </div>

                {/* Main Content */}
                <div className='flex-1 p-4 h-1/3'>
                    <div className='h-full w-full mx-auto grid grid-cols-1 lg:grid-cols-2 gap-4'>
                        {/* Video Section */}
                        <div className='bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden flex flex-col'>
                            <div className='flex-shrink-0 p-3 border-b border-gray-200 dark:border-gray-700'>
                                <div className='flex items-center gap-2'>
                                    <FileVideo className='h-4 w-4 text-blue-600 dark:text-blue-400' />
                                    <h3 className='font-semibold text-sm text-gray-800 dark:text-gray-200'>
                                        Analyzed Video
                                    </h3>
                                </div>
                                <p className='text-xs text-gray-600 dark:text-gray-400 mt-1 truncate'>
                                    {videoFile.name}
                                </p>
                            </div>
                            <div className='flex-1 p-3 flex items-center justify-center'>
                                <video
                                    src={url ? url : ''}
                                    controls
                                    className="w-full max-h-[62%] object-contain rounded-lg"
                                />
                            </div>
                        </div>

                        {/* Analysis Section */}
                        <div className='bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden flex flex-col'>
                            <div className='flex-shrink-0 p-3 border-b border-gray-200 dark:border-gray-700'>
                                <div className='flex items-center gap-2'>
                                    <MessageSquare className='h-4 w-4 text-blue-600 dark:text-blue-400' />
                                    <h3 className='font-semibold text-sm text-gray-800 dark:text-gray-200'>
                                        Form Analysis & Recommendations
                                    </h3>
                                </div>
                            </div>
                            <div className='flex-1 p-3 overflow-y-auto'>
                                {analysis ? (
                                    <div className='h-full'>
                                        <p className='text-l text-gray-700 dark:text-gray-300 leading-relaxed'>
                                            {analysis}
                                        </p>
                                    </div>
                                ) : (
                                    <div className='h-full flex flex-col items-center justify-center'>
                                        <Target className='h-8 w-8 text-gray-400 mb-2' />
                                        <p className='text-xs text-gray-500 dark:text-gray-400 text-center'>
                                            AI analysis will appear here once available
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                <div className='flex-shrink-0 flex flex-row w-full justify-end px-6 py-4 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm border-t border-gray-200 dark:border-gray-700'>
                <Button 
                    onClick={handleReturn} 
                    disabled={!saved}
                    size="lg"
                    className="h-12 px-6 text-sm font-semibold transition-all duration-300 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white border-0 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-lg"
                >
                    Return Home
                </Button>
                </div>
            </div>
        )
    )
}