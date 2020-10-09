(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("d50762db-9ad3-481e-942e-dbe7762772f6");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'd50762db-9ad3-481e-942e-dbe7762772f6' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"10f20d9f-b546-4225-be13-e2a0a8ebd108":{"roots":{"references":[{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5354","type":"PolyAnnotation"},{"attributes":{"data_source":{"id":"5369"},"glyph":{"id":"5370"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5371"},"selection_glyph":null,"view":{"id":"5373"}},"id":"5372","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"5353"}},"id":"5347","type":"BoxZoomTool"},{"attributes":{"data_source":{"id":"5364"},"glyph":{"id":"5365"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5366"},"selection_glyph":null,"view":{"id":"5368"}},"id":"5367","type":"GlyphRenderer"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5366","type":"Patch"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5353","type":"BoxAnnotation"},{"attributes":{"data":{"x":{"__ndarray__":"MCYUyCgqBcBwqLaSWRAFwPGs+ye73ATAcbFAvRypBMDytYVSfnUEwHO6yuffQQTA9L4PfUEOBMB1w1QSo9oDwPbHmacEpwPAdszePGZzA8D30CPSxz8DwHjVaGcpDAPA+dmt/IrYAsB63vKR7KQCwPriNydOcQLAe+d8vK89AsD868FREQoCwH3wBudy1gHA/vRLfNSiAcB/+ZARNm8BwAD+1aaXOwHAgAIbPPkHAcABB2DRWtQAwIILpWa8oADAAxDq+x1tAMCEFC+RfzkAwAQZdCbhBQDACjtyd4Wk/78MRPyhSD3/vw5NhswL1v6/D1YQ985u/r8RX5ohkgf+vxJoJExVoP2/FHGudhg5/b8Wejih29H8vxiDwsueavy/GYxM9mED/L8aldYgJZz7vxyeYEvoNPu/HqfqdavN+r8gsHSgbmb6vyG5/sox//m/I8KI9fSX+b8lyxIguDD5vybUnEp7yfi/KN0mdT5i+L8p5rCfAfv3vyvvOsrEk/e/LfjE9Ics978uAU8fS8X2vzAK2UkOXva/MhNjdNH29b8zHO2elI/1vzUld8lXKPW/Ny4B9BrB9L84N4se3ln0vzpAFUmh8vO/PEmfc2SL8789UimeJyTzvz9bs8jqvPK/QGQ9861V8r9Cbccdce7xv0R2UUg0h/G/RX/bcvcf8b9HiGWdurjwv0mR78d9UfC/lDTz5IHU77+YRgc6CAbvv5xYG4+ON+6/nmov5BRp7b+ifEM5m5rsv6SOV44hzOu/qKBr46f96r+ssn84Li/qv67Ek420YOm/stan4jqS6L+26Ls3wcPnv7j6z4xH9ea/vAzk4c0m5r/AHvg2VFjlv8QwDIzaieS/xEIg4WC747/IVDQ25+ziv8xmSIttHuK/0Hhc4PNP4b/UinA1eoHgv7A5CRUBZt+/sF0xvw3J3b+4gVlpGizcv8ClgRMnj9q/yMmpvTPy2L/Q7dFnQFXXv9AR+hFNuNW/2DUivFkb1L/gWUpmZn7Sv+h9chBz4dC/4EM1df+Izr/wi4XJGE/Lv/DT1R0yFci/ABwmckvbxL8QZHbGZKHBv0BYjTX8zry/YOgt3i5btr/A8JwNw86vvwAR3l4o56K/AMV8wDb+h78Aun76M6CLP0COXq2nz6M/ALcOLqFbsD8AJ26Fbs+2P+CWzdw7Q70/YIMWmoTbwT9QO8ZFaxXFP0DzdfFRT8g/MKslnTiJyz8wY9VIH8POP5CNQvqC/tA/iGkaUHab0j+ARfKlaTjUP3ghyvtc1dU/eP2hUVBy1z9w2XmnQw/ZP2i1Uf02rNo/YJEpUypJ3D9YbQGpHebdP1BJ2f4Qg98/qJJYKgKQ4D+kgETVe17hP6BuMID1LOI/nFwcK2/74j+YSgjW6MnjP5g49IBimOQ/lCbgK9xm5T+QFMzWVTXmP4wCuIHPA+c/iPCjLEnS5z+E3o/XwqDoP4TMe4I8b+k/gLpnLbY96j98qFPYLwzrP3iWP4Op2us/dIQrLiOp7D90chfZnHftP3BgA4QWRu4/bE7vLpAU7z9oPNvZCePvPzKVY8LBWPA/MIzZl/6/8D8wg09tOyfxPy56xUJ4jvE/LHE7GLX18T8qaLHt8VzyPyhfJ8MuxPI/JladmGsr8z8mTRNuqJLzPyREiUPl+fM/Ijv/GCJh9D8gMnXuXsj0Px4p68ObL/U/HiBhmdiW9T8aF9duFf71PxoOTURSZfY/GgXDGY/M9j8W/DjvyzP3PxbzrsQIm/c/EuokmkUC+D8S4Zpvgmn4PxLYEEW/0Pg/Ds+GGvw3+T8OxvzvOJ/5Pwq9csV1Bvo/CrTomrJt+j8Gq15w79T6Pwai1EUsPPs/BplKG2mj+z8CkMDwpQr8PwKHNsbicfw//n2smx/Z/D/+dCJxXED9P/5rmEaZp/0/+mIOHNYO/j/6WYTxEnb+P/ZQ+sZP3f4/9kdwnIxE/z/2PuZxyav/P/kariODCQBAeRZpjiE9AED3EST5v3AAQHcN32NepABA9QiazvzXAEB1BFU5mwsBQPX/D6Q5PwFAc/vKDthyAUDz9oV5dqYBQHHyQOQU2gFA8e37TrMNAkBx6ba5UUECQO/kcSTwdAJAb+Asj46oAkDu2+f5LNwCQO7b5/ks3AJAb+Asj46oAkDv5HEk8HQCQHHptrlRQQJA8e37TrMNAkBx8kDkFNoBQPP2hXl2pgFAc/vKDthyAUD1/w+kOT8BQHUEVTmbCwFA9QiazvzXAEB3Dd9jXqQAQPcRJPm/cABAeRZpjiE9AED5Gq4jgwkAQPY+5nHJq/8/9kdwnIxE/z/2UPrGT93+P/pZhPESdv4/+mIOHNYO/j/+a5hGmaf9P/50InFcQP0//n2smx/Z/D8ChzbG4nH8PwKQwPClCvw/BplKG2mj+z8GotRFLDz7PwarXnDv1Po/CrTomrJt+j8KvXLFdQb6Pw7G/O84n/k/Ds+GGvw3+T8S2BBFv9D4PxLhmm+Cafg/EuokmkUC+D8W867ECJv3Pxb8OO/LM/c/GgXDGY/M9j8aDk1EUmX2PxoX124V/vU/HiBhmdiW9T8eKevDmy/1PyAyde5eyPQ/Ijv/GCJh9D8kRIlD5fnzPyZNE26okvM/JladmGsr8z8oXyfDLsTyPypose3xXPI/LHE7GLX18T8uesVCeI7xPzCDT207J/E/MIzZl/6/8D8ylWPCwVjwP2g829kJ4+8/bE7vLpAU7z9wYAOEFkbuP3RyF9mcd+0/dIQrLiOp7D94lj+DqdrrP3yoU9gvDOs/gLpnLbY96j+EzHuCPG/pP4Tej9fCoOg/iPCjLEnS5z+MAriBzwPnP5AUzNZVNeY/lCbgK9xm5T+YOPSAYpjkP5hKCNboyeM/nFwcK2/74j+gbjCA9SziP6SARNV7XuE/qJJYKgKQ4D9QSdn+EIPfP1htAakd5t0/YJEpUypJ3D9otVH9NqzaP3DZeadDD9k/eP2hUVBy1z94Icr7XNXVP4BF8qVpONQ/iGkaUHab0j+QjUL6gv7QPzBj1Ugfw84/MKslnTiJyz9A83XxUU/IP1A7xkVrFcU/YIMWmoTbwT/gls3cO0O9PwAnboVuz7Y/ALcOLqFbsD9Ajl6tp8+jPwC6fvozoIs/AMV8wDb+h78AEd5eKOeiv8DwnA3Dzq+/YOgt3i5btr9AWI01/M68vxBkdsZkocG/ABwmckvbxL/w09UdMhXIv/CLhckYT8u/4EM1df+Izr/ofXIQc+HQv+BZSmZmftK/2DUivFkb1L/QEfoRTbjVv9Dt0WdAVde/yMmpvTPy2L/ApYETJ4/av7iBWWkaLNy/sF0xvw3J3b+wOQkVAWbfv9SKcDV6geC/0Hhc4PNP4b/MZkiLbR7iv8hUNDbn7OK/xEIg4WC747/EMAyM2onkv8Ae+DZUWOW/vAzk4c0m5r+4+s+MR/Xmv7bouzfBw+e/stan4jqS6L+uxJONtGDpv6yyfzguL+q/qKBr46f96r+kjleOIczrv6J8Qzmbmuy/nmov5BRp7b+cWBuPjjfuv5hGBzoIBu+/lDTz5IHU779Jke/HfVHwv0eIZZ26uPC/RX/bcvcf8b9EdlFINIfxv0Jtxx1x7vG/QGQ9861V8r8/W7PI6rzyvz1SKZ4nJPO/PEmfc2SL8786QBVJofLzvzg3ix7eWfS/Ny4B9BrB9L81JXfJVyj1vzMc7Z6Uj/W/MhNjdNH29b8wCtlJDl72vy4BTx9Lxfa/LfjE9Ics978r7zrKxJP3vynmsJ8B+/e/KN0mdT5i+L8m1JxKe8n4vyXLEiC4MPm/I8KI9fSX+b8huf7KMf/5vyCwdKBuZvq/HqfqdavN+r8cnmBL6DT7vxqV1iAlnPu/GYxM9mED/L8Yg8LLnmr8vxZ6OKHb0fy/FHGudhg5/b8SaCRMVaD9vxFfmiGSB/6/D1YQ985u/r8OTYbMC9b+vwxE/KFIPf+/Cjtyd4Wk/78EGXQm4QUAwIQUL5F/OQDAAxDq+x1tAMCCC6VmvKAAwAEHYNFa1ADAgAIbPPkHAcAA/tWmlzsBwH/5kBE2bwHA/vRLfNSiAcB98AbnctYBwPzrwVERCgLAe+d8vK89AsD64jcnTnECwHre8pHspALA+dmt/IrYAsB41WhnKQwDwPfQI9LHPwPAdszePGZzA8D2x5mnBKcDwHXDVBKj2gPA9L4PfUEOBMBzusrn30EEwPK1hVJ+dQTAcbFAvRypBMDxrPsnu9wEwHCotpJZEAXAMCYUyCgqBcA=","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5381"},"selection_policy":{"id":"5380"}},"id":"5364","type":"ColumnDataSource"},{"attributes":{},"id":"5345","type":"ResetTool"},{"attributes":{},"id":"5382","type":"UnionRenderers"},{"attributes":{"below":[{"id":"5337"}],"center":[{"id":"5340"},{"id":"5344"}],"left":[{"id":"5341"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5367"},{"id":"5372"}],"title":{"id":"5374"},"toolbar":{"id":"5355"},"toolbar_location":"above","x_range":{"id":"5329"},"x_scale":{"id":"5333"},"y_range":{"id":"5331"},"y_scale":{"id":"5335"}},"id":"5328","subtype":"Figure","type":"Plot"},{"attributes":{"axis":{"id":"5341"},"dimension":1,"ticker":null},"id":"5344","type":"Grid"},{"attributes":{},"id":"5342","type":"BasicTicker"},{"attributes":{"overlay":{"id":"5354"}},"id":"5349","type":"LassoSelectTool"},{"attributes":{},"id":"5350","type":"UndoTool"},{"attributes":{},"id":"5338","type":"BasicTicker"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5365","type":"Patch"},{"attributes":{"source":{"id":"5364"}},"id":"5368","type":"CDSView"},{"attributes":{"data":{"x":{"__ndarray__":"76Nx/fdDBcDf/9H8KZMAwCTLjkXFEQDA4QQYoE93+79uUvicthP7v6mYA8enOfq/eUYpBFt4+L9WPSX9iS73vwyIBjeKUfa/SWuIieML879mHJOe3+rxvy1xqwaXWPG/EHHyYNMe8L868kr+SATwv8ZgU+JrO++/PVcJABFU7r9d0WW+6bPtv6GwPqWZu+y/shPjZMld67+DfTSQumTnv1TuZkyF6ea/Y4Wr85J65L8qdui2f1fjv2Ursx8V7eK/FLEojO3c4r+eG1hsqQLcv6piGvh+UNu/UZ/ns5fC2r/ROTPro7PZv/V6UolmrNm/uiMZFv+U2L+SvK3pl2/Yv7HPBvhUMtS/gN6WGmfb0r/7HPSav1jSv6Yc991M29G/B9bi5NdJ0L93eCvUExvLv2ZGeX7vyMe/LEY+0sxXxL/vzhJ2v1O/v6IfV8Odari/Nfavt+Ygor9DujW00quSvw+vSr5w+XO/4L6Ypb51Vj/YV8X6DhuiP6PDRdFlQrg/qyxmmzM/uj+MgxGLXanIP9vd62SZns0/cQhUOGZLzz8D84ibPwPSP+g/bYjaJtI/xzYnCQem1D9/vkyAmbHUPyu3+eJhwtQ/WevDmH5m1T/BWFm1YtDWP8ubuj2gn9s/N7CP/idD3D+fUvsVAYLeP/6xC85NDt8/LkR3aOkj4D/przguDCLhPzdlAv52p+E/eqtSfEPV4T9QVQzYmJriP5TqnjmkrOQ/Co8tD2+i5T/6hC/lJu3lP1nGuytbn+Y/ljDpm1uI6T+lcnqL5vzpP4dNIk6of+o/7KuR+nsv6z9Aida8X9LrPwAGizkAhew/mDBhppBb7T+Et5XF17rtP48f5ASNxe4/NpRD/JHs7j8MYHKMQbjwP1O3Rn9CD/E/UuQZDxBi8T86Mmqi9lPyPzbB1aCJ7fI/lzPEMXJ08z966B4Hevb1Pz3nuT+sE/Y/YYOVLd049j/BWEuLxr/2Px/g1ZL2APk/9w/F2ic2+j+S4BD+W7f+P1ln95l1uwBAeeC8fY5kAUBkmsfPUYABQHsKftI4XQJA7tvn+SzcAkA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"IrgcBRB45T9CAFwGrNnuP7hp4nR13O8/kP3zL1hE8j/J1oOxJHbyP6wzfhws4/I/xFzrfdLD8z9VYW0Bu2j0P/q7fOQ61/Q/XMo7Ow569j/NcbYwkAr3P2pHqny0U/c/eMeGT5bw9z/jhtqA2/33P84nawclMfg/Mar9v/tq+D+pi2aQBZP4P9hTsJYZ0fg/FDvHpo0o+T+f4PJb0Sb6P2tE5qyeRfo/px4VQ1vh+j924kUSICr7Pyc1E7i6RPs/u9P1nMRI+z+M/HTSqn/8P6uz/CDwlfw/FgyDCa2n/D/GmJmCi8n8P6Gw1S5zyvw/ids8HWDt/D9uSMoCDfL8Pwom/2C1ef0/MCStHJOk/T9hfKEM6LT9P2scQWSWxP0/P6VjA8X2/T95SL3CTk7+P5prGAhxg/4/nRvcMoO6/j+JaU8EYgX/PwNH5RGrPP8/J0AhZXy3/z+LlJdaqNr/P6jaoEcD9v8/jFnqW2cBAECwivUdNiQAQA8XRZcJYQBAs5htzvxoAEAcjFjsSsUAQO9eJ8v07ABARKDCMVv6AEAwj7j5MyABQP7ThqhtIgFAbHOScGBKAUDoywSYGUsBQHObLx4mTAFAtj6M6WdWAUCMlVUrBm0BQL2p2wP6uQFAA/vofzLEAUAqtV8RIOgBQCC74Nzk8AFAhugOLX0EAkD9FceFQSQCQKdMwN/uNAJAb1WKb6g6AkCqigEbU1MCQFLdM4eUlQJA4bHl4U20AkCf8KXcpL0CQMt4d2Xr0wJAEyZ9cwsxA0BVTm/RnD8DQLFJxAn1TwNAfjVSf+9lA0Ao0Zr3S3oDQMBgMQegkANAEybMFHKrA0DwtrL4WrcDQPKDnKCx2ANAh3KIP5LdA0ADmBxjEC4EQNWt0Z/QQwRAFHnGA4RYBECOjJqo/ZQEQE5wNWhiuwRA5gxxjBzdBEAeuseBnn0FQM957g/rhAVA2GBlSzeOBUAw1tKi8a8FQAh4taQ9QAZA/kOx9omNBkAkOIT/1q0HQKyz+8y6XQhAPHDePkeyCEAyzePnKMAIQD4FP2mcLglA9+3zfBZuCUA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5383"},"selection_policy":{"id":"5382"}},"id":"5369","type":"ColumnDataSource"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5370","type":"Line"},{"attributes":{"callback":null},"id":"5352","type":"HoverTool"},{"attributes":{},"id":"5351","type":"SaveTool"},{"attributes":{},"id":"5331","type":"DataRange1d"},{"attributes":{},"id":"5329","type":"DataRange1d"},{"attributes":{},"id":"5346","type":"PanTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5345"},{"id":"5346"},{"id":"5347"},{"id":"5348"},{"id":"5349"},{"id":"5350"},{"id":"5351"},{"id":"5352"}]},"id":"5355","type":"Toolbar"},{"attributes":{"axis":{"id":"5337"},"ticker":null},"id":"5340","type":"Grid"},{"attributes":{},"id":"5383","type":"Selection"},{"attributes":{"formatter":{"id":"5378"},"ticker":{"id":"5338"}},"id":"5337","type":"LinearAxis"},{"attributes":{"source":{"id":"5369"}},"id":"5373","type":"CDSView"},{"attributes":{},"id":"5381","type":"Selection"},{"attributes":{},"id":"5376","type":"BasicTickFormatter"},{"attributes":{},"id":"5348","type":"WheelZoomTool"},{"attributes":{},"id":"5380","type":"UnionRenderers"},{"attributes":{},"id":"5335","type":"LinearScale"},{"attributes":{},"id":"5333","type":"LinearScale"},{"attributes":{"text":""},"id":"5374","type":"Title"},{"attributes":{"formatter":{"id":"5376"},"ticker":{"id":"5342"}},"id":"5341","type":"LinearAxis"},{"attributes":{},"id":"5378","type":"BasicTickFormatter"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5371","type":"Line"}],"root_ids":["5328"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"10f20d9f-b546-4225-be13-e2a0a8ebd108","root_ids":["5328"],"roots":{"5328":"d50762db-9ad3-481e-942e-dbe7762772f6"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();