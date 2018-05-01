//
//  ImageSavingViewController.swift
//  Pastichio
//
//  Created by Jamie Williamson on 01/05/2018.
//  Copyright © 2018 Jamie Williamson. All rights reserved.
//

import UIKit

class ImageSavingViewController: UIViewController {

    //MARK: Properties
    @IBOutlet weak var imageView: UIImageView!
    var styledImage: UIImage?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        imageView.image = styledImage
        // Do any additional setup after loading the view.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    @IBAction func saveImage(_ sender: UIButton) {
        UIImageWriteToSavedPhotosAlbum(styledImage!, self, nil, nil)
    }
}
