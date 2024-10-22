'use client'
import { useEffect, useState } from "react";

export default function Home() {
    const [message, setMessage] = useState("");
    const [selectedImage, setSelectedImage] = useState(null);
    const [predictedImage, setPredictedImage] = useState("");

    useEffect(() => {
        const fetchMessage = async () => {
            const response = await fetch("http://localhost:3002/message");
            const data = await response.json();
            setMessage(data.message);
        };

        fetchMessage();
    }, []);

    const handleImageChange = (e) => {
        setSelectedImage(e.target.files[0]);
    };

    const handleImageDrop = (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) {
            setSelectedImage(file);
        }
    };

    const handleImageUpload = async (e) => {
        e.preventDefault();

        if (!selectedImage) {
            alert("Please, select an image.");
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedImage);

        const response = await fetch("http://localhost:3002/predict/image/", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);
            setPredictedImage(imageUrl);
        } else {
            alert("Error al procesar la imagen.");
        }
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
             <div className="bg-white shadow-md rounded-lg p-6 mb-4 max-w-xl w-full">
             <h1 className="text-2xl font-bold text-center text-gray-700">{message}</h1>
            </div>
            <div className="bg-white shadow-md rounded-lg p-6 max-w-xl w-full">    
                <h2 className="text-xl text-center text-gray-600 mb-6">Upload your image</h2> 
                <form onSubmit={handleImageUpload} className="flex flex-col items-center">
                    <label
                        className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-blue-800 rounded-lg cursor-pointer hover:bg-blue-50 transition duration-200"
                        onDragOver={(e) => e.preventDefault()}  // Prevent default behavior to allow dropping
                        onDrop={handleImageDrop}  // Handle the drop event
                    >
                        <input
                            type="file"
                            accept="image/*"
                            onChange={handleImageChange}
                            className="hidden" // Hide the default input
                        />
                        <span className="flex flex-col items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-blue-800" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M10 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V10l-8-6zm2 2l5 4h-3v6h-4v-6H5l5-4z"/>
                        </svg>
                            <span className="mt-2 text-gray-600">Drag and drop an image here, or click to upload.</span>
                        </span>
                    </label>

                    {/* Preview Selected Image */}
                    {selectedImage && (
                        <div className="mt-4">
                            <h2 className="text-lg font-semibold text-gray-700">Image Preview:</h2>
                            <img 
                                src={URL.createObjectURL(selectedImage)} 
                                alt="Imagen seleccionada" 
                                className="mt-2 rounded shadow-md max-h-48" 
                            />
                        </div>
                    )}
                    
                    <button
                        type="submit"
                        className="bg-blue-800 text-white font-semibold py-2 px-4 rounded shadow hover:bg-yellow-200 hover:text-black transition duration-200 mt-4"
                    >
                        Predict Image
                    </button>
                </form>

                {predictedImage && (
                    <div className="mt-6 text-center">
                        <h2 className="text-lg font-semibold text-gray-700">Processed Image:</h2>
                        <img src={predictedImage} alt="Imagen procesada" className="mt-2 rounded shadow-md" />
                    </div>
                )}
            </div>
        </div>
    );
}
