import { DeleteIcon, Download, File } from 'lucide-react';
import { useEffect, useState } from 'react';
import { NavLink  } from 'react-router-dom';
import { deleteFile, getAllFiles } from '../helper/db';
import toast from 'react-hot-toast';

const RecentFiles = () => {
    const [generatedFiles, setRecentFiles] = useState(null)

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
    const handleRedownload = (data) => {
        if (!data) {
            console.error("No saved data found to redownload.");
            return;
        }

        const { fileData, filename = "download.glb" } = JSON.parse(data);
        const blob = base64ToBlob(fileData);
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', filename);
        document.body.appendChild(link);
        link.click();
        link.parentNode.removeChild(link);
        URL.revokeObjectURL(url); // Clean up the created URL
    };

    const handleRemove = (id) => {
        const confirmed = window.confirm("Are you sure you want to delete this?");
        if (confirmed) {
            try {
                deleteFile(id)
                setRecentFiles(generatedFiles.filter(item => item.id !== id))
                toast.success('Sucessfully Removed! 🗑️')
            } catch (error) {
                
            }
        }
    }

    useEffect(()=> {
        
       (async () => {
            const glbFiles = await getAllFiles();
            setRecentFiles(glbFiles);
        })();
    },[])

    return (
        <div className="space-y-6 pt-5">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-white mb-3">Recent Generations</h2>
              <p className="text-slate-300 text-lg">Download your previously generated 3D models</p>
            </div>

            { generatedFiles?.length > 0 && <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-slate-500/30 mx-3">
              <div className="p-6 border-b border-slate-500/20">
                <h3 className="text-xl font-semibold text-white">Generated Files</h3>
              </div>
                <div className="divide-y divide-slate-500/20">
                    {generatedFiles.map((file) => (
                    <div key={file.id} className="p-6 flex items-center justify-between hover:bg-slate-500/5 transition-colors">
                        <div className="flex items-center space-x-4">
                        <div className="w-12 h-12 bg-gradient-to-r from-slate-500 to-pink-500 rounded-lg flex items-center justify-center">
                            <Download className="w-6 h-6 text-white" />
                        </div>
                        <div className='w-32 sm:w-64 md:w-11/12 truncate'>
                            <h4 className="text-white font-medium text-sm md:text-lg">{file.filename}</h4>
                        </div>
                        </div>
                        <div className='grid md:grid-cols-2 md:space-x-3'>
                            <button 
                                onClick={() => handleRedownload(file)}
                                className="bg-slate-600 hover:bg-slate-700 text-white px-6 py-2 rounded-lg font-medium transition-colors">
                                <Download className="w-3 h-3 text-white md:hidden block" />
                                <span className='hidden md:block'>Download</span>
                            </button>
                            <button 
                                onClick={() => handleRemove(file.id)}
                                className="bg-red-400 hover:bg-red-500 text-white px-6 py-2 rounded-lg font-medium transition-colors">
                                <DeleteIcon className="w-3 h-3 text-white md:hidden block" />
                                <span className='hidden md:block'>Delete</span>
                            </button>
                        </div>
                    </div>
                    ))}
                </div>
            </div>}
            { generatedFiles?.length === 0 &&
                <div className='justify-items-center space-y-3'>
                    <File className='items-center' size={"15%"} />
                    <h4 className="text-white font-medium text-lg">{"No recent files"}</h4>
                    <NavLink 
                        to="/"
                        style={{
                            textDecoration: "none",
                            color:"white"
                        }}
                        className="px-4 py-2 rounded-md text-sm font-medium transition-all bg-slate-600 text-white shadow-lg no-underline hover:shadow-2xl hover:bg-slate-700"
                    >Generate Now</NavLink>
                </div>
            }
        </div>
    )
}

export default RecentFiles;