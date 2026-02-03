import { Download, File, Trash2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { deleteFile, getAllFiles } from '../helper/db';
import toast from 'react-hot-toast';

const RecentFiles = () => {
    const [generatedFiles, setRecentFiles] = useState(null);

    const base64ToBlob = (base64Data, contentType = 'model/gltf-binary') => {
        const byteCharacters = atob(base64Data.split(',')[1]);
        const byteArrays = [];
        for (let offset = 0; offset < byteCharacters.length; offset += 512) {
            const slice = byteCharacters.slice(offset, offset + 512);
            const byteNumbers = new Array(slice.length);
            for (let i = 0; i < slice.length; i++) {
                byteNumbers[i] = slice.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            byteArrays.push(byteArray);
        }
        return new Blob(byteArrays, { type: contentType });
    };

    const handleRedownload = (file) => {
        if (!file) {
            console.error("No saved data found to redownload.");
            return;
        }

        const { fileData, filename = "download.glb" } = file;
        const blob = base64ToBlob(fileData);
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', filename);
        document.body.appendChild(link);
        link.click();
        link.parentNode.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const handleRemove = (id) => {
        const confirmed = window.confirm("Are you sure you want to delete this?");
        if (confirmed) {
            try {
                deleteFile(id);
                setRecentFiles(generatedFiles.filter(item => item.id !== id));
                toast.success('Successfully removed!');
            } catch (error) {
                toast.error('Failed to delete file');
            }
        }
    };

    useEffect(() => {
        (async () => {
            const glbFiles = await getAllFiles();
            setRecentFiles(glbFiles);
        })();
    }, []);

    return (
        <div className="space-y-8">
            {/* Header */}
            <section className="text-center">
                <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-white">
                    Download your previously generated 3D models
                </h1>
            </section>

            {/* Files List */}
            {generatedFiles?.length > 0 && (
                <section className="glass overflow-hidden">
                    <div className="p-5 border-b border-white/10">
                        <h3 className="text-lg font-semibold text-white text-center">Generated Files</h3>
                    </div>
                    <div className="divide-y divide-white/10">
                        {generatedFiles.map((file) => (
                            <div 
                                key={file.id} 
                                className="p-5 flex items-center justify-between hover:bg-white/[0.02] transition-colors"
                            >
                                <div className="flex items-center gap-4">
                                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 via-violet-500 to-pink-500 flex items-center justify-center">
                                        <Download className="w-5 h-5 text-white" />
                                    </div>
                                    <div className="max-w-[200px] sm:max-w-md truncate">
                                        <span className="text-white font-medium">{file.filename}</span>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <button 
                                        onClick={() => handleRedownload(file)}
                                        className="flex items-center gap-2 rounded-xl bg-white/10 hover:bg-white/15 px-4 py-2 text-sm font-medium text-white transition"
                                    >
                                        <Download className="w-4 h-4" />
                                        <span className="hidden sm:inline">Download</span>
                                    </button>
                                    <button 
                                        onClick={() => handleRemove(file.id)}
                                        className="flex items-center gap-2 rounded-xl bg-red-500/20 hover:bg-red-500/30 px-4 py-2 text-sm font-medium text-red-300 transition"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                        <span className="hidden sm:inline">Delete</span>
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>
            )}

            {/* Empty State */}
            {generatedFiles?.length === 0 && (
                <section className="glass p-12 text-center">
                    <File className="w-16 h-16 text-white/30 mx-auto mb-4" />
                    <h4 className="text-xl font-medium text-white mb-2">No recent files</h4>
                    <p className="text-white/60 mb-6">Generate your first 3D model to see it here</p>
                    <NavLink 
                        to="/"
                        className="inline-block rounded-xl bg-white/10 hover:bg-white/15 px-6 py-3 text-sm font-medium text-white transition"
                    >
                        Generate Now
                    </NavLink>
                </section>
            )}

            {/* Loading State */}
            {generatedFiles === null && (
                <section className="glass p-12 text-center">
                    <div className="w-8 h-8 border-2 border-white/20 border-t-white rounded-full animate-spin mx-auto mb-4" />
                    <p className="text-white/60">Loading files...</p>
                </section>
            )}
        </div>
    );
};

export default RecentFiles;