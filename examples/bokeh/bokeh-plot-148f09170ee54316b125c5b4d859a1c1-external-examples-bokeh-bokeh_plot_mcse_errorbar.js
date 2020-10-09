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
    
      
      
    
      var element = document.getElementById("4a764117-ca1e-41a5-9898-7fb32922cbec");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '4a764117-ca1e-41a5-9898-7fb32922cbec' but no matching script tag was found.")
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
                    
                  var docs_json = '{"1b259daf-c5d5-4a60-accf-477abc20f5ba":{"roots":{"references":[{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"22168","type":"Dash"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"22156","type":"PolyAnnotation"},{"attributes":{"axis_label":"Value $\\\\pm$ MCSE for quantiles","formatter":{"id":"22180"},"ticker":{"id":"22144"}},"id":"22143","type":"LinearAxis"},{"attributes":{"axis":{"id":"22139"},"ticker":null},"id":"22142","type":"Grid"},{"attributes":{},"id":"22135","type":"LinearScale"},{"attributes":{"callback":null},"id":"22154","type":"HoverTool"},{"attributes":{},"id":"22137","type":"LinearScale"},{"attributes":{},"id":"22140","type":"BasicTicker"},{"attributes":{"axis":{"id":"22143"},"dimension":1,"ticker":null},"id":"22146","type":"Grid"},{"attributes":{"children":[[{"id":"22130"},0,0]]},"id":"22190","type":"GridBox"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"22182"},"ticker":{"id":"22140"}},"id":"22139","type":"LinearAxis"},{"attributes":{},"id":"22144","type":"BasicTicker"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"22167","type":"Dash"},{"attributes":{},"id":"22148","type":"PanTool"},{"attributes":{"overlay":{"id":"22155"}},"id":"22149","type":"BoxZoomTool"},{"attributes":{},"id":"22147","type":"ResetTool"},{"attributes":{},"id":"22153","type":"SaveTool"},{"attributes":{"overlay":{"id":"22156"}},"id":"22151","type":"LassoSelectTool"},{"attributes":{"children":[{"id":"22192"},{"id":"22190"}]},"id":"22193","type":"Column"},{"attributes":{},"id":"22187","type":"UnionRenderers"},{"attributes":{},"id":"22188","type":"Selection"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"22147"},{"id":"22148"},{"id":"22149"},{"id":"22150"},{"id":"22151"},{"id":"22152"},{"id":"22153"},{"id":"22154"}]},"id":"22157","type":"Toolbar"},{"attributes":{},"id":"22133","type":"DataRange1d"},{"attributes":{"source":{"id":"22166"}},"id":"22170","type":"CDSView"},{"attributes":{},"id":"22180","type":"BasicTickFormatter"},{"attributes":{"data_source":{"id":"22171"},"glyph":{"id":"22172"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"22173"},"selection_glyph":null,"view":{"id":"22175"}},"id":"22174","type":"GlyphRenderer"},{"attributes":{"data":{"xs":[[0.05,0.05],[0.09736842105263158,0.09736842105263158],[0.14473684210526316,0.14473684210526316],[0.19210526315789472,0.19210526315789472],[0.23947368421052628,0.23947368421052628],[0.28684210526315784,0.28684210526315784],[0.33421052631578946,0.33421052631578946],[0.381578947368421,0.381578947368421],[0.4289473684210526,0.4289473684210526],[0.47631578947368414,0.47631578947368414],[0.5236842105263158,0.5236842105263158],[0.5710526315789474,0.5710526315789474],[0.618421052631579,0.618421052631579],[0.6657894736842105,0.6657894736842105],[0.7131578947368421,0.7131578947368421],[0.7605263157894736,0.7605263157894736],[0.8078947368421052,0.8078947368421052],[0.8552631578947368,0.8552631578947368],[0.9026315789473683,0.9026315789473683],[0.95,0.95]],"ys":[[0.06542507260989384,0.07779740932825195],[0.08783828101486021,0.09606135597316004],[0.09938050528645676,0.10552315572600299],[0.10794861576931342,0.11348270780810846],[0.11602734321885094,0.12203033330056018],[0.12350211909750375,0.12787167939093885],[0.12891616867169003,0.13323376159544686],[0.13473004088066556,0.13958486303584752],[0.14096481555257187,0.1460531576744907],[0.14714684275886325,0.15141900007039433],[0.15249024001585332,0.15717175454660753],[0.15761692093229293,0.16322637365277673],[0.16410729163047846,0.16867037944462424],[0.17025660267247006,0.17489182297617661],[0.17657098829958856,0.18216819265595885],[0.185369502560243,0.18948857007780892],[0.19264880841245133,0.19711830100588604],[0.20027922695307082,0.2044362824100993],[0.2124901683329975,0.2184324622519046],[0.23108929611368745,0.2369230522137875]]},"selected":{"id":"22188"},"selection_policy":{"id":"22187"}},"id":"22171","type":"ColumnDataSource"},{"attributes":{},"id":"22131","type":"DataRange1d"},{"attributes":{},"id":"22182","type":"BasicTickFormatter"},{"attributes":{"line_color":{"value":"#1f77b4"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"22172","type":"MultiLine"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"+Pb8QR1Vsj+MDk73BYq3PyGJcH1IOro/R5qJktxXvD8QN5uHrHi+P+EQ5SCCFsA/nVhJgBDHwD+XHUgZYI7BP8j2F6KAXsI/quCp6LMbwz/9bt2KgNHDP6iPWFCyiMQ/uNKGtzpMxT8aURts6RbGP6Hdfy6V9cY/RM+ntqz9xz+OhgC/8fHIP+aFOeHb5sk//PpdgzyUyz/cM4wQ6vPNPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"22186"},"selection_policy":{"id":"22185"}},"id":"22166","type":"ColumnDataSource"},{"attributes":{"source":{"id":"22171"}},"id":"22175","type":"CDSView"},{"attributes":{"toolbars":[{"id":"22157"}],"tools":[{"id":"22147"},{"id":"22148"},{"id":"22149"},{"id":"22150"},{"id":"22151"},{"id":"22152"},{"id":"22153"},{"id":"22154"}]},"id":"22191","type":"ProxyToolbar"},{"attributes":{"toolbar":{"id":"22191"},"toolbar_location":"above"},"id":"22192","type":"ToolbarBox"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"22155","type":"BoxAnnotation"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"22173","type":"MultiLine"},{"attributes":{},"id":"22185","type":"UnionRenderers"},{"attributes":{"below":[{"id":"22139"}],"center":[{"id":"22142"},{"id":"22146"}],"left":[{"id":"22143"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"22169"},{"id":"22174"}],"title":{"id":"22176"},"toolbar":{"id":"22157"},"toolbar_location":null,"x_range":{"id":"22131"},"x_scale":{"id":"22135"},"y_range":{"id":"22133"},"y_scale":{"id":"22137"}},"id":"22130","subtype":"Figure","type":"Plot"},{"attributes":{"text":"sigma_a"},"id":"22176","type":"Title"},{"attributes":{},"id":"22186","type":"Selection"},{"attributes":{},"id":"22150","type":"WheelZoomTool"},{"attributes":{"data_source":{"id":"22166"},"glyph":{"id":"22167"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"22168"},"selection_glyph":null,"view":{"id":"22170"}},"id":"22169","type":"GlyphRenderer"},{"attributes":{},"id":"22152","type":"UndoTool"}],"root_ids":["22193"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"1b259daf-c5d5-4a60-accf-477abc20f5ba","root_ids":["22193"],"roots":{"22193":"4a764117-ca1e-41a5-9898-7fb32922cbec"}}];
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