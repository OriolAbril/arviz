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
    
      
      
    
      var element = document.getElementById("3cfab6e4-c78d-4934-9ac5-cbccff0e87a7");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '3cfab6e4-c78d-4934-9ac5-cbccff0e87a7' but no matching script tag was found.")
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
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
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
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
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
                    
                  var docs_json = '{"0ce6bf96-7265-455c-b309-58305ef8f97a":{"roots":{"references":[{"attributes":{},"id":"5239","type":"DataRange1d"},{"attributes":{},"id":"5241","type":"LinearScale"},{"attributes":{},"id":"5246","type":"BasicTicker"},{"attributes":{"formatter":{"id":"5285"},"ticker":{"id":"5250"}},"id":"5249","type":"LinearAxis"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5262","type":"PolyAnnotation"},{"attributes":{"axis":{"id":"5245"},"ticker":null},"id":"5248","type":"Grid"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5278","type":"Line"},{"attributes":{},"id":"5237","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5273","type":"Patch"},{"attributes":{},"id":"5256","type":"WheelZoomTool"},{"attributes":{},"id":"5290","type":"Selection"},{"attributes":{},"id":"5291","type":"UnionRenderers"},{"attributes":{},"id":"5250","type":"BasicTicker"},{"attributes":{},"id":"5243","type":"LinearScale"},{"attributes":{"callback":null},"id":"5260","type":"HoverTool"},{"attributes":{},"id":"5253","type":"ResetTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5253"},{"id":"5254"},{"id":"5255"},{"id":"5256"},{"id":"5257"},{"id":"5258"},{"id":"5259"},{"id":"5260"}]},"id":"5263","type":"Toolbar"},{"attributes":{"below":[{"id":"5245"}],"center":[{"id":"5248"},{"id":"5252"}],"left":[{"id":"5249"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5275"},{"id":"5280"}],"title":{"id":"5282"},"toolbar":{"id":"5263"},"toolbar_location":"above","x_range":{"id":"5237"},"x_scale":{"id":"5241"},"y_range":{"id":"5239"},"y_scale":{"id":"5243"}},"id":"5236","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"5285","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"5277"}},"id":"5281","type":"CDSView"},{"attributes":{"overlay":{"id":"5262"}},"id":"5257","type":"LassoSelectTool"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5290"},"selection_policy":{"id":"5291"}},"id":"5272","type":"ColumnDataSource"},{"attributes":{},"id":"5292","type":"Selection"},{"attributes":{},"id":"5293","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5279","type":"Line"},{"attributes":{"data_source":{"id":"5277"},"glyph":{"id":"5278"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5279"},"selection_glyph":null,"view":{"id":"5281"}},"id":"5280","type":"GlyphRenderer"},{"attributes":{"formatter":{"id":"5287"},"ticker":{"id":"5246"}},"id":"5245","type":"LinearAxis"},{"attributes":{"overlay":{"id":"5261"}},"id":"5255","type":"BoxZoomTool"},{"attributes":{},"id":"5258","type":"UndoTool"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5274","type":"Patch"},{"attributes":{"text":""},"id":"5282","type":"Title"},{"attributes":{},"id":"5259","type":"SaveTool"},{"attributes":{"data":{"x":{"__ndarray__":"AIhrx0qhAMBTpdPy30oAwIcV3pSKSADAeksmJCsk/7/W/4KrIe79v/J5EkN5Lv2/6Rm/ewBj/L9HBshalZ/7v4eetOqw1Pq/M0FBid1/+L/MxTir3UL4v2NGTULl0vG/vglA1LCP8b9MhloAJHvxv4jMybF7O/G/1cr2LH9i8L8pgaBrwR/vvxjdPQgSue6/oiOCjWZu7r/XS2ZaQiLuv3rH1fpPSOy/O0qfrZDv679ywZJV7nvrv3cE4a1Qpeq/mq2PCwd+6r8nWcg3IDPpvxlpKwSSW+e/DkK51NTy5r/CoAB5QhHkvxzHtYAgN+O/lloxLjf94r+hEHuhE5jiv6lNhX/dSeK/XGXUKnWM4b+ZU7BTqm7hvyD1yOkapdu/R1mfhP3K2b8QRyi6hqjZv+vU7KTIG9m/c9g3HkZm2L8i3hX+Jp7Uv3jjZubiTtG/G5g7bE7g0L+gtHTxxsbLv8qgw9OSPsO/wk3Fli6Lv793r5qNi2G6v+4fpJXOx7e/JSNt+Cmyor+wpGMzOZWPv+NgH+0V+p8/rKMYdmtKoT+ijzi3CoClP6HFzcN3+aw/An6wZJC2rj/fjXcv+8izPya3VjXdR74/31j4aPnAvj+YVwPSH5jLP0Rhlq2+xdE/Z+8OEShc0j/HbYunn/7SPztkEGRGAdM/zc6RIRz11j8ecfL0+LvZP3NVYEqv8ts/Mwa3bG2z3T/6BQpUBxbgPyHtyoWeIeA/1TsnLb8y4j/I5DVpp7TiP6rJoJmqAuM/H6l65oAC5D8iNe1EAebkP9jgMiKqDeU/rfxygvH75T+mmvhYgA7mPxZ0TALSD+Y/+Mdunfk06D+QTPzgFejoP7Jbt8V0LOk/HXtbAxSC6z+X+ZB1ecLrP7rfxLlfOfA/EQIqugi18D+wtja4PPPwP8cMMHkC+/E/4D+bA+Lm8j9TxVciwB7zP/Nu2/y8YvM/0bVQX8138z9Ps6ncu5L3P7D3zkBG1Pc/3Xx4Wtn3+D81sz751n75PwP2E0EQzfo/FmnNNPpT/D9HvYM0u3r8P0bQS0ui6/0/OMPZY8PtAUA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"APAocWq97j9atVgaQGrvP/LUQ9bqbu8/Q9rsbept8D8VgD4q7wjxPwfDdl7DaPE/DHMgwn/O8T/c/JtSNTDyP7ywpYqnlfI/Zl9fOxHA8z8anWMqkd7zP85c2V6NFvc/IfvflSc49z/avNL/bUL3P7wZGydCYvc/lpqEacDO9z+23xelDzj4P7qI8H27Ufg/GHefXGZk+D8KbWZpb3f4PyKOSgHs7fg/cS2Y1BsE+T+kT5tqBCH5P+K+h9SrVvk/mhQcPX5g+T+26Q3yN7P5P7ol9X4bKfo/fK/RykpD+j/Q179hr/v6PzmO0t83Mvs/WqlzNLJA+z/YO6EX+1n7P5asHqCIbfs/qeZKteKc+z8a6xNrVaT7P1zhxqJci/w/1xRsT6DG/D8e97oo78r8P2NlYuuG3Pw/8gQ5PDfz/D88RD0gO2z9P5EjM6Mj1v0//Yx4Mvbj/T+2tOiQk0P+P/PFw9IWzP4/ktVJi6YD/z+EKpOj8yz/PwHfUovBQf8/c0seWDe1/z9bnMzGauD/P2Ef7RX6HwBARzHs1pQiAEAfcW4VACsAQIubh+/yOQBA/GDJIG09AEA33r3sI08AQN1a1XQfeQBAY+Gj5QN7AEC9GpD+wNwAQBRm2epbHAFA9u4QgcIlAUDctnj66S8BQEQGQWYUMAFA7RwZwlFvAUASJ0+Pv5sBQFcFpvQqvwFAY3DL1jbbAUC/QIHqwAICQKRdudAzBAJAe+ek5VdGAkCZvCbtlFYCQDUZNFNVYAJAJFXPHFCAAkCkpp0owJwCQBtcRkS1oQJAll9OMH6/AkBVEx8L0MECQIOOSUD6wQJA/9itM58GA0CSiR+8Ah0DQHbrtpiOJQNAZG9rgEJwA0AzH7IuT3gDQO43ce5XDgRAhICKLkItBECsrQ0uzzwEQDIDTJ7AfgRA+M/mgLi5BEBV8ZUIsMcEQL3bNj+v2ARAdC3UV/PdBEDUbCr3ruQFQOy9M5AR9QVANx+eVvY9BkDNrE++tV8GQIH9RBBEswZARlozjf4UB0BS7yDNrh4HQBL00pLoegdAnOHsseH2CEA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5292"},"selection_policy":{"id":"5293"}},"id":"5277","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"5272"},"glyph":{"id":"5273"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5274"},"selection_glyph":null,"view":{"id":"5276"}},"id":"5275","type":"GlyphRenderer"},{"attributes":{},"id":"5254","type":"PanTool"},{"attributes":{},"id":"5287","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5261","type":"BoxAnnotation"},{"attributes":{"axis":{"id":"5249"},"dimension":1,"ticker":null},"id":"5252","type":"Grid"},{"attributes":{"source":{"id":"5272"}},"id":"5276","type":"CDSView"}],"root_ids":["5236"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"0ce6bf96-7265-455c-b309-58305ef8f97a","root_ids":["5236"],"roots":{"5236":"3cfab6e4-c78d-4934-9ac5-cbccff0e87a7"}}];
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