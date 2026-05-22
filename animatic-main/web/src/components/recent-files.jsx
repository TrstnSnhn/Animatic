import { Download, Eye, File, Trash2, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { deleteFile, getAllFiles } from '../helper/db';
import toast from 'react-hot-toast';
import GlbPreview from './glb-preview';
import { dataUrlToBlob } from '../utils/glb-file';

const formatSavedAt = (value) => {
    if (!value) return 'Saved date unavailable';

    const savedAt = new Date(value);
    if (Number.isNaN(savedAt.getTime())) {
        return 'Saved date unavailable';
    }

    return new Intl.DateTimeFormat(undefined, {
        dateStyle: 'medium',
        timeStyle: 'short',
    }).format(savedAt);
};

const RecentFiles = () => {
    const [generatedFiles, setRecentFiles] = useState(null);
    const [deletingId, setDeletingId] = useState(null);
    const [listMessage, setListMessage] = useState(null);
    const [previewFile, setPreviewFile] = useState(null);
    const previewUrlRef = useRef(null);

    const revokePreviewUrl = () => {
        if (previewUrlRef.current) {
            URL.revokeObjectURL(previewUrlRef.current);
            previewUrlRef.current = null;
        }
    };

    const clearPreview = () => {
        revokePreviewUrl();
        setPreviewFile(null);
    };

    useEffect(() => revokePreviewUrl, []);

    const handleRedownload = (file) => {
        if (!file) {
            console.error("No saved data found to redownload.");
            return;
        }

        const { fileData, filename = "download.glb" } = file;
        let blob;
        try {
            blob = dataUrlToBlob(fileData);
        } catch (error) {
            console.error("Failed to prepare saved GLB download:", error);
            setListMessage({ type: 'error', text: 'This saved GLB could not be prepared for download.' });
            toast.error('Saved GLB could not be downloaded.');
            return;
        }
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        try {
            link.href = url;
            link.setAttribute('download', filename);
            document.body.appendChild(link);
            link.click();
        } finally {
            link.parentNode?.removeChild(link);
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        }
    };

    const handlePreview = (file) => {
        if (!file) return;

        try {
            const blob = dataUrlToBlob(file.fileData);
            const url = URL.createObjectURL(blob);
            revokePreviewUrl();
            previewUrlRef.current = url;
            setPreviewFile({
                id: file.id,
                filename: file.filename || 'download.glb',
                url,
            });
            setListMessage(null);
        } catch (error) {
            console.error("Failed to prepare saved GLB preview:", error);
            setListMessage({ type: 'error', text: 'This saved GLB could not be opened in the preview.' });
            toast.error('Saved GLB could not be previewed.');
        }
    };

    const handleRemove = async (id) => {
        const confirmed = window.confirm("Delete this saved GLB from this browser?");
        if (confirmed) {
            setDeletingId(id);
            try {
                await deleteFile(id);
                setRecentFiles((files) => files.filter(item => item.id !== id));
                if (previewFile?.id === id) {
                    clearPreview();
                }
                setListMessage({ type: 'success', text: 'Saved GLB deleted from this browser.' });
                toast.success('Successfully removed!');
            } catch (error) {
                console.error("Failed to delete file:", error);
                setListMessage({ type: 'error', text: 'Could not delete the saved GLB. Please try again.' });
                toast.error('Failed to delete file. Please try again.');
            } finally {
                setDeletingId(null);
            }
        }
    };

    useEffect(() => {
        (async () => {
            try {
                const glbFiles = await getAllFiles();
                setRecentFiles(glbFiles);
            } catch (error) {
                console.error("Failed to load recent files:", error);
                toast.error('Failed to load recent files.');
                setListMessage({ type: 'error', text: 'Recent files could not be loaded. Refresh the page or try again.' });
                setRecentFiles([]);
            }
        })();
    }, []);

    return (
        <div className="space-y-8">
            {/* Header */}
            <section className="text-center">
                <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-white">
                    Download your previously generated GLB files
                </h1>
            </section>

            {listMessage && (
                <section
                    className={`rounded-xl border px-4 py-3 text-sm ${
                        listMessage.type === 'error'
                            ? 'border-red-400/30 bg-red-500/10 text-red-100'
                            : 'border-emerald-400/30 bg-emerald-500/10 text-emerald-100'
                    }`}
                    role={listMessage.type === 'error' ? 'alert' : 'status'}
                    aria-live={listMessage.type === 'error' ? 'assertive' : 'polite'}
                >
                    {listMessage.text}
                </section>
            )}

            {previewFile && (
                <GlbPreview
                    src={previewFile.url}
                    filename={previewFile.filename}
                    helperText="Preview this saved GLB from local browser storage. Download and delete controls remain available below."
                    actions={
                        <button
                            type="button"
                            onClick={clearPreview}
                            className="inline-flex items-center gap-2 rounded-xl bg-white/10 hover:bg-white/15 px-4 py-2 text-sm font-medium text-white transition"
                            aria-label={`Close preview for ${previewFile.filename}`}
                        >
                            <X className="h-4 w-4" aria-hidden="true" />
                            <span>Close Preview</span>
                        </button>
                    }
                />
            )}

            {/* Files List */}
            {generatedFiles?.length > 0 && (
                <section className="glass overflow-hidden">
                    <div className="p-5 border-b border-white/10">
                        <h3 className="text-lg font-semibold text-white text-center">Generated GLB Files</h3>
                    </div>
                    <div className="divide-y divide-white/10">
                        {generatedFiles.map((file) => (
                            <div 
                                key={file.id} 
                                className="p-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between hover:bg-white/[0.02] transition-colors"
                            >
                                <div className="flex items-center gap-4">
                                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 via-violet-500 to-pink-500 flex items-center justify-center">
                                        <Download className="w-5 h-5 text-white" aria-hidden="true" />
                                    </div>
                                    <div className="max-w-[200px] sm:max-w-md truncate">
                                        <span className="block truncate text-white font-medium" title={file.filename}>
                                            {file.filename}
                                        </span>
                                        <span className="mt-1 block text-xs text-white/50">
                                            {formatSavedAt(file.createdAt)}
                                        </span>
                                    </div>
                                </div>
                                <div className="grid grid-cols-3 gap-2 sm:flex sm:items-center">
                                    <button
                                        type="button"
                                        onClick={() => handlePreview(file)}
                                        aria-label={`Preview ${file.filename || 'saved GLB'}`}
                                        className="flex items-center justify-center gap-2 rounded-xl bg-white/10 hover:bg-white/15 px-3 py-2 text-sm font-medium text-white transition sm:px-4"
                                    >
                                        <Eye className="w-4 h-4" aria-hidden="true" />
                                        <span className="hidden sm:inline">Preview</span>
                                    </button>
                                    <button 
                                        type="button"
                                        onClick={() => handleRedownload(file)}
                                        aria-label={`Download ${file.filename || 'saved GLB'}`}
                                        className="flex items-center justify-center gap-2 rounded-xl bg-white/10 hover:bg-white/15 px-3 py-2 text-sm font-medium text-white transition sm:px-4"
                                    >
                                        <Download className="w-4 h-4" aria-hidden="true" />
                                        <span className="hidden sm:inline">Download</span>
                                    </button>
                                    <button 
                                        type="button"
                                        onClick={() => handleRemove(file.id)}
                                        disabled={deletingId === file.id}
                                        aria-label={`Delete ${file.filename || 'saved GLB'}`}
                                        className="flex items-center justify-center gap-2 rounded-xl bg-red-500/20 hover:bg-red-500/30 disabled:cursor-not-allowed disabled:opacity-60 px-3 py-2 text-sm font-medium text-red-300 transition sm:px-4"
                                    >
                                        <Trash2 className="w-4 h-4" aria-hidden="true" />
                                        <span className="hidden sm:inline">
                                            {deletingId === file.id ? "Deleting..." : "Delete"}
                                        </span>
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
                    <h4 className="text-xl font-medium text-white mb-2">No generated GLB files yet</h4>
                    <p className="text-white/60 mb-6">
                        Successful generations are saved locally in this browser so you can download them again later.
                    </p>
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
                <section className="glass p-12 text-center" role="status" aria-live="polite">
                    <div className="w-8 h-8 border-2 border-white/20 border-t-white rounded-full animate-spin mx-auto mb-4" />
                    <p className="text-white/60">Loading recent GLB files...</p>
                </section>
            )}
        </div>
    );
};

export default RecentFiles;
