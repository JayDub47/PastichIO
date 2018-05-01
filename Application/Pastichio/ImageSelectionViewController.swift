//
//  ViewController.swift
//  Pastichio
//
//  Created by Jamie Williamson on 17/02/2018.
//  Copyright © 2018 Jamie Williamson. All rights reserved.
//

import UIKit
import Vision
import CoreML

class ImageSelectionViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    //MARK: Properties
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var stylizeImageButton: UIButton!
    
    var outputImage: UIImage?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.imageView.image = UIImage(named: "defaultImage")
        self.stylizeImageButton.isEnabled = false
        // Do any additional setup after loading the view, typically from a nib.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        //Retrieve the image that has been selected
        guard let image: UIImage = info[UIImagePickerControllerOriginalImage] as? UIImage else {
            fatalError("Error with media selection")
        }
        //Set image in ImageViewer with resizing
        self.imageView.image = self.resizeImage(image)
        
        //Enable stylization button now an image has been selected
        self.stylizeImageButton.isEnabled = true
        
        dismiss(animated: true, completion: nil)
    }
    
    func resizeImage(_ image: UIImage) -> UIImage! {
        let newSize = CGSize(width: 512, height: 512)
        let rectangle = CGRect(x: 0, y: 0, width: 512, height: 512)
        
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        image.draw(in: rectangle)
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return newImage
    }
    
    func stylizationResultsMethod(request: VNRequest, error: Error?) {
        print("Image Stylized")
        
        //Collect the results as a pixel buffer
        guard let results = request.results as? [VNPixelBufferObservation] else {
            fatalError("Unexpected Error when converting stylization output")
        }
        
        let result = results[0]
        let output = result.pixelBuffer
        
        let outputCIImage = CIImage.init(cvImageBuffer: output)
        outputImage = UIImage.init(ciImage: outputCIImage)
        

    }
    
    //MARK: Actions
    func stylizeImage() {
        print("Beginning Image Stylization")
        do {
            let inputImage = self.imageView.image?.cgImage
            let model = try VNCoreMLModel(for: coreml_network().model)
            let request = VNCoreMLRequest(model: model, completionHandler: stylizationResultsMethod)
            let handler = VNImageRequestHandler(cgImage: inputImage!)
            try handler.perform([request])
        } catch {
            print("Unexpected error occured during stylization: \(error)" )
        }
    }
    
    @IBAction func takePhoto(_ sender: UIButton) {
        let imagePickerController = UIImagePickerController()
        imagePickerController.delegate = self
        imagePickerController.sourceType = .camera
        present(imagePickerController, animated: true, completion: nil)
    }
    
    @IBAction func photoLibrary(_ sender: UIButton) {
        let imagePickerController = UIImagePickerController()
        imagePickerController.delegate = self
        imagePickerController.sourceType = .photoLibrary
        present(imagePickerController, animated: true, completion: nil)
    }
    
    //MARK: Naviation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        
        self.stylizeImage()
        let destination = segue.destination as! ImageSavingViewController
        destination.styledImage = outputImage
    }
    
}

